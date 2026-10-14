from pathlib import Path
from pathlib import Path
import json
import uuid
from datetime import datetime, UTC

from app.ai.director import create_video_plan, save_video_plan


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(r"C:\MascotAI")

JOB_DIR = BASE_DIR / "temp" / "jobs"


# ============================================================
# LTX DEFAULT SETTINGS
# ============================================================

DEFAULT_LTX_SETTINGS = {
    "duration": 2,
    "num_frames": 49,
    "aspect_ratio": "16:9",
    "quality": "Low",
    "motion_mode": "Normal",
    "audio_mode": "None",
    "narration_text": "",
    "seed": 1234,
    "steps": 4,
    "guidance_scale": 1.0,
}


# ============================================================
# PRODUCTION PIPELINE
# ============================================================

class ProductionPipeline:
    """
    Creates an AI video production job.

    Current stage:

        User prompt
             â†“
        NVIDIA AI Director
             â†“
        Production plan
             â†“
        LTX scene preparation

    IMPORTANT:
    LTX generation is NOT started here.

    This stage only prepares all files required by the
    video-generation worker.
    """

    def __init__(self, base_dir: Path = BASE_DIR):

        self.base_dir = Path(base_dir)

        self.job_dir = (
            self.base_dir
            / "temp"
            / "jobs"
        )

        self.job_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ========================================================
    # CREATE PRODUCTION JOB
    # ========================================================

    def create_job(
        self,
        user_prompt: str,
        duration: int = 10,
        aspect_ratio: str = "9:16",
        quality: str = "standard",
    ):

        user_prompt = user_prompt.strip()

        if not user_prompt:
            raise ValueError(
                "Video prompt cannot be empty."
            )

        if duration <= 0:
            raise ValueError(
                "Video duration must be greater than zero."
            )

        job_id = str(uuid.uuid4())

        job_path = (
            self.job_dir
            / job_id
        )

        job_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # JOB METADATA
        # ----------------------------------------------------

        job = {
            "job_id": job_id,
            "status": "planning",
            "created_at": (
                datetime.now(UTC).isoformat()
            ),
            "user_prompt": user_prompt,
            "duration": duration,
            "aspect_ratio": aspect_ratio,
            "quality": quality,
            "generation_started": False,
            "generation_completed": False,
            "ltx_generation_started": False,
        }

        self._save_json(
            job_path / "job.json",
            job,
        )

        # ----------------------------------------------------
        # AI DIRECTOR
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("AI DIRECTOR")
        print("=" * 70)

        print("Prompt:")
        print(user_prompt)

        print()
        print(
            "Requested duration:",
            duration,
        )

        print(
            "Aspect ratio:",
            aspect_ratio,
        )

        print()

        plan = create_video_plan(
            user_prompt=user_prompt,
            duration=duration,
            aspect_ratio=aspect_ratio,
        )

        # ----------------------------------------------------
        # SAVE PRODUCTION PLAN
        # ----------------------------------------------------

        plan_path = (
            job_path
            / "video_plan.json"
        )

        save_video_plan(
            plan,
            plan_path,
        )

        # ----------------------------------------------------
        # CREATE SCENES
        # ----------------------------------------------------

        scenes = []

        for scene in plan["scenes"]:

            scene_id = int(
                scene["scene_id"]
            )

            scene_duration = int(
                scene["duration_seconds"]
            )

            scene_title = scene.get(
                "title",
                f"Scene {scene_id}",
            )

            visual_prompt = scene.get(
                "visual_prompt",
                "",
            ).strip()

            # ------------------------------------------------
            # SCENE DIRECTORY
            # ------------------------------------------------

            scene_path = (
                job_path
                / f"scene_{scene_id}"
            )

            scene_path.mkdir(
                parents=True,
                exist_ok=True,
            )

            # ------------------------------------------------
            # LTX PROMPT
            # ------------------------------------------------

            prompt_file = (
                scene_path
                / "prompt.txt"
            )

            with open(
                prompt_file,
                "w",
                encoding="utf-8",
            ) as f:

                f.write(
                    visual_prompt
                )

            # ------------------------------------------------
            # LTX SETTINGS
            # ------------------------------------------------

            ltx_settings = (
                DEFAULT_LTX_SETTINGS.copy()
            )

            ltx_settings[
                "duration"
            ] = scene_duration

            ltx_settings[
                "aspect_ratio"
            ] = aspect_ratio

            # Convert the platform's portrait ratio
            # into a useful LTX generation ratio.

            if aspect_ratio == "9:16":
                ltx_settings[
                    "aspect_ratio"
                ] = "9:16"

            elif aspect_ratio == "16:9":
                ltx_settings[
                    "aspect_ratio"
                ] = "16:9"

            elif aspect_ratio == "1:1":
                ltx_settings[
                    "aspect_ratio"
                ] = "1:1"

            settings_file = (
                scene_path
                / "settings.json"
            )

            self._save_json(
                settings_file,
                ltx_settings,
            )

            # ------------------------------------------------
            # SCENE METADATA
            # ------------------------------------------------

            scene_data = {
                "scene_id": scene_id,
                "scene_number": scene.get(
                    "scene_number",
                    scene_id,
                ),
                "title": scene_title,
                "duration_seconds": scene_duration,
                "visual_prompt": visual_prompt,
                "camera": scene.get(
                    "camera",
                    "",
                ),
                "action": scene.get(
                    "action",
                    "",
                ),
                "lighting": scene.get(
                    "lighting",
                    "",
                ),
                "dialogue": scene.get(
                    "dialogue",
                    "",
                ),
                "voiceover": scene.get(
                    "voiceover",
                    "",
                ),
                "music": scene.get(
                    "music",
                    "",
                ),
                "sound_effects": scene.get(
                    "sound_effects",
                    [],
                ),
                "visual_effects": scene.get(
                    "visual_effects",
                    [],
                ),
                "status": "prepared",
                "video_file": None,
                "prompt_file": str(
                    prompt_file
                ),
                "settings_file": str(
                    settings_file
                ),
            }

            scene_json = (
                scene_path
                / "scene.json"
            )

            self._save_json(
                scene_json,
                scene_data,
            )

            scenes.append(
                scene_data
            )

        # ----------------------------------------------------
        # VALIDATE TOTAL DURATION
        # ----------------------------------------------------

        total_duration = sum(
            int(
                scene[
                    "duration_seconds"
                ]
            )
            for scene in scenes
        )

        if total_duration != duration:

            raise ValueError(
                "Scene duration mismatch: "
                f"expected {duration}, "
                f"got {total_duration}."
            )

        # ----------------------------------------------------
        # UPDATE JOB
        # ----------------------------------------------------

        job["status"] = "prepared"

        job[
            "generation_started"
        ] = False

        job[
            "generation_completed"
        ] = False

        job[
            "ltx_generation_started"
        ] = False

        job[
            "scene_count"
        ] = len(scenes)

        job[
            "total_scene_duration"
        ] = total_duration

        job[
            "plan_file"
        ] = str(plan_path)

        job[
            "scenes"
        ] = scenes

        self._save_json(
            job_path / "job.json",
            job,
        )

        # ----------------------------------------------------
        # PRINT SUMMARY
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("PRODUCTION PLAN CREATED")
        print("=" * 70)

        print()
        print("Job ID:")
        print(job_id)

        print()
        print("Scenes:")

        for scene in scenes:

            print(
                f"  Scene {scene['scene_id']}: "
                f"{scene['duration_seconds']} sec"
            )

            print(
                f"    {scene['title']}"
            )

            print(
                "    LTX files prepared: YES"
            )

        print()
        print(
            "Total duration:",
            total_duration,
            "seconds",
        )

        print()
        print("Plan:")
        print(plan_path)

        print()
        print("Job folder:")
        print(job_path)

        print()
        print("LTX GENERATION:")
        print("  NOT STARTED")

        print()
        print(
            "GPU generation is intentionally disabled "
            "for this test."
        )

        print("=" * 70)

        return {
            "job_id": job_id,
            "job_path": str(job_path),
            "plan_path": str(plan_path),
            "status": "prepared",
            "scene_count": len(scenes),
            "duration": total_duration,
            "ltx_generation_started": False,
            "scenes": scenes,
        }

    # ========================================================
    # LOAD JOB
    # ========================================================

    def load_job(
        self,
        job_id: str,
    ):

        job_path = (
            self.job_dir
            / job_id
            / "job.json"
        )

        if not job_path.exists():

            raise FileNotFoundError(
                f"Job not found: {job_id}"
            )

        with open(
            job_path,
            "r",
            encoding="utf-8",
        ) as f:

            return json.load(f)

    # ========================================================
    # SAVE JSON
    # ========================================================

    @staticmethod
    def _save_json(
        path: Path,
        data: dict,
    ):

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                data,
                f,
                indent=2,
                ensure_ascii=False,
            )


# ============================================================
# SIMPLE FUNCTION API
# ============================================================

def create_production_job(
    user_prompt: str,
    duration: int = 10,
    aspect_ratio: str = "9:16",
    quality: str = "standard",
):

    pipeline = ProductionPipeline()

    return pipeline.create_job(
        user_prompt=user_prompt,
        duration=duration,
        aspect_ratio=aspect_ratio,
        quality=quality,
    )


# ============================================================
# COMMAND-LINE TEST
# ============================================================

if __name__ == "__main__":

    result = create_production_job(
        user_prompt=(
            "A futuristic robot walking through "
            "a neon city at night"
        ),
        duration=12,
        aspect_ratio="9:16",
        quality="standard",
    )

    print()
    print("JOB RESULT")

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )
# ============================================================
