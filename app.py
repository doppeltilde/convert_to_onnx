import logging
import os
import subprocess
import sys
import tempfile
import tarfile
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.request import urlopen, urlretrieve

import streamlit as st
from huggingface_hub import HfApi, whoami

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Application configuration."""

    hf_token: str
    hf_username: str
    is_using_user_token: bool
    transformers_version: str = "3.6.1"
    hf_base_url: str = "https://huggingface.co"
    transformers_base_url: str = (
        "https://github.com/huggingface/transformers.js/archive/refs"
    )
    repo_path: Path = Path("./transformers.js")

    @classmethod
    def from_env(cls) -> "Config":
        """Create config from environment variables and secrets."""
        system_token = st.secrets.get("HF_TOKEN")
        user_token = st.session_state.get("user_hf_token")

        if user_token:
            hf_username = whoami(token=user_token)["name"]
        else:
            hf_username = (
                os.getenv("SPACE_AUTHOR_NAME") or whoami(token=system_token)["name"]
            )

        hf_token = user_token or system_token

        if not hf_token:
            raise ValueError(
                "When the user token is not provided, the system token must be set."
            )

        return cls(
            hf_token=hf_token,
            hf_username=hf_username,
            is_using_user_token=bool(user_token),
        )


class ModelConverter:
    """Handles model conversion and upload operations."""

    def __init__(self, config: Config):
        self.config = config
        self.api = HfApi(token=config.hf_token)

    def _get_ref_type(self) -> str:
        """Determine the reference type for the transformers repository."""
        url = f"{self.config.transformers_base_url}/tags/{self.config.transformers_version}.tar.gz"
        try:
            return "tags" if urlopen(url).getcode() == 200 else "heads"
        except Exception as e:
            logger.warning(f"Failed to check tags, defaulting to heads: {e}")
            return "heads"

    def setup_repository(self) -> None:
        """Download and setup transformers repository if needed."""
        if self.config.repo_path.exists():
            return

        ref_type = self._get_ref_type()
        archive_url = f"{self.config.transformers_base_url}/{ref_type}/{self.config.transformers_version}.tar.gz"
        archive_path = Path(f"./transformers_{self.config.transformers_version}.tar.gz")

        try:
            urlretrieve(archive_url, archive_path)
            self._extract_archive(archive_path)
            logger.info("Repository downloaded and extracted successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to setup repository: {e}")
        finally:
            archive_path.unlink(missing_ok=True)

    def _extract_archive(self, archive_path: Path) -> None:
        """Extract the downloaded archive."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(tmp_dir)

            extracted_folder = next(Path(tmp_dir).iterdir())
            extracted_folder.rename(self.config.repo_path)

    def _run_conversion_subprocess(
        self, input_model_id: str, extra_args: List[str] = None
    ) -> subprocess.CompletedProcess:
        """Run the conversion subprocess with the given arguments."""
        cmd = [
            sys.executable,
            "-m",
            "scripts.convert",
            "--quantize",
            "--model_id",
            input_model_id,
        ]

        if extra_args:
            cmd.extend(extra_args)

        return subprocess.run(
            cmd,
            cwd=self.config.repo_path,
            capture_output=True,
            text=True,
            env={
                "HF_TOKEN": self.config.hf_token,
            },
        )

    def convert_model(
        self, input_model_id: str, trust_remote_code=False
    ) -> Tuple[bool, Optional[str]]:
        """Convert the model to ONNX format."""
        try:
            if trust_remote_code:
                if not self.config.is_using_user_token:
                    raise Exception(
                        "Trust Remote Code requires your own HuggingFace token."
                    )

                result = self._run_conversion_subprocess(
                    input_model_id, extra_args=["--trust_remote_code"]
                )
            else:
                result = self._run_conversion_subprocess(input_model_id)

            if result.returncode != 0:
                return False, result.stderr

            return True, result.stderr

        except Exception as e:
            return False, str(e)

    def upload_model(self, input_model_id: str, output_model_id: str) -> Optional[str]:
        """Upload the converted model to Hugging Face."""
        model_folder_path = self.config.repo_path / "models" / input_model_id

        try:
            self.api.create_repo(output_model_id, exist_ok=True, private=False)

            readme_path = f"{model_folder_path}/README.md"

            if not os.path.exists(readme_path):
                with open(readme_path, "w") as file:
                    file.write(self.generate_readme(input_model_id))

            self.api.upload_folder(
                folder_path=str(model_folder_path), repo_id=output_model_id
            )
            return None
        except Exception as e:
            return str(e)
        finally:
            shutil.rmtree(model_folder_path, ignore_errors=True)

    def generate_readme(self, imi: str):
        return (
            "---\n"
            "library_name: transformers.js\n"
            "base_model:\n"
            f"- {imi}\n"
            "---\n\n"
            f"# {imi.split('/')[-1]} (ONNX)\n\n"
            f"This is an ONNX version of [{imi}](https://huggingface.co/{imi}). "
            "It was automatically converted and uploaded using "
            "[this space](https://huggingface.co/spaces/onnx-community/convert-to-onnx).\n"
        )


def main():
    """Main application entry point."""
    st.write("## Convert a Hugging Face model to ONNX")

    try:
        config = Config.from_env()
        converter = ModelConverter(config)
        converter.setup_repository()

        input_model_id = st.text_input(
            "Enter the Hugging Face model ID to convert. Example: `EleutherAI/pythia-14m`"
        )

        if not input_model_id:
            return

        st.text_input(
            f"Optional: Your Hugging Face write token. Fill it if you want to upload the model under your account.",
            type="password",
            key="user_hf_token",
        )
        trust_remote_code = st.toggle("Optional: Trust Remote Code.")
        if trust_remote_code:
            st.warning(
                "This option should only be enabled for repositories you trust and in which you have read the code, as it will execute arbitrary code present in the model repository. When this option is enabled, you must use your own Hugging Face write token."
            )

        if config.hf_username == input_model_id.split("/")[0]:
            same_repo = st.checkbox(
                "Do you want to upload the ONNX weights to the same repository?"
            )
        else:
            same_repo = False

        model_name = input_model_id.split("/")[-1]

        output_model_id = f"{config.hf_username}/{model_name}"

        if not same_repo:
            output_model_id += "-ONNX"

        output_model_url = f"{config.hf_base_url}/{output_model_id}"

        if not same_repo and converter.api.repo_exists(output_model_id):
            st.write("This model has already been converted! 🎉")
            st.link_button(f"Go to {output_model_id}", output_model_url, type="primary")
            return

        st.write(f"URL where the model will be converted and uploaded to:")
        st.code(output_model_url, language="plaintext")

        if not st.button(label="Proceed", type="primary"):
            return

        with st.spinner("Converting model..."):
            success, stderr = converter.convert_model(
                input_model_id, trust_remote_code=trust_remote_code
            )
            if not success:
                st.error(f"Conversion failed: {stderr}")
                return

            st.success("Conversion successful!")
            st.code(stderr)

        with st.spinner("Uploading model..."):
            error = converter.upload_model(input_model_id, output_model_id)
            if error:
                st.error(f"Upload failed: {error}")
                return

            st.success("Upload successful!")
            st.write("You can now go and view the model on Hugging Face!")
            st.link_button(f"Go to {output_model_id}", output_model_url, type="primary")

    except Exception as e:
        logger.exception("Application error")
        st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
