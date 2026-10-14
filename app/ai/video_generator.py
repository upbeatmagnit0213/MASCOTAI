from pathlib import Path
import subprocess


class VideoGenerator:
    """
    Provider-independent video generation interface.

    Currently uses the local LTX-Video worker.
    Later we can add cloud providers without changing
    the rest of the production pipeline.
    """

    def __init__(
        self,
        python_path: Path,
        worker_path: Path,
        base_dir: Path,
    ):
        self.python_path = Path(python_path)
        self.worker_path = Path(worker_path)
        self.base_dir = Path(base_dir)

    def generate_scene(
        self,
        scene_id: int,
        prompt: str,
        output_file: Path,
        settings_file: Path,
        job_id: str,
        prompt_file: Path,
    ):
        """
        Generate one video scene using the local LTX worker.
        """

        output_file.parent.mkdir(parents=True, exist_ok=True)

        command = [
            str(self.python_path),
            str(self.worker_path),
            job_id,
            str(prompt_file),
            str(output_file),
            str(settings_file),
        ]

        process = subprocess.Popen(
            command,
            cwd=str(self.base_dir),
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )

        return {
            "scene_id": scene_id,
            "process_id": process.pid,
            "output_file": str(output_file),
            "status": "started",
        }