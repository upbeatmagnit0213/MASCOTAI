import json
from pathlib import Path


def create_storyboard(video_plan: dict) -> dict:
    """
    Converts the AI Director plan into a more detailed storyboard.

    This is intentionally provider-independent.
    Later the AI model will create much richer scenes automatically.
    """

    prompt = video_plan["original_prompt"]
    duration = video_plan["duration"]
    aspect_ratio = video_plan["aspect_ratio"]

    scenes = video_plan.get("scenes", [])

    storyboard = {
        "version": "1.0",
        "prompt": prompt,
        "duration": duration,
        "aspect_ratio": aspect_ratio,
        "scenes": []
    }

    for scene in scenes:
        scene_id = scene["scene_id"]

        if scene_id == 1:
            visual = (
                f"Opening shot based on: {prompt}. "
                "Establish the environment, characters and situation."
            )
            action = "Introduce the main character and begin the story."
            camera = "cinematic establishing shot"

        elif scene_id == len(scenes):
            visual = (
                f"Final scene based on: {prompt}. "
                "Create a satisfying and visually memorable ending."
            )
            action = "Complete the story with a clear ending."
            camera = "cinematic closing shot"

        else:
            visual = (
                f"Continue the story: {prompt}. "
                "Develop the characters and action naturally."
            )
            action = "Continue the main action and move the story forward."
            camera = "dynamic cinematic camera movement"

        storyboard["scenes"].append({
            "scene_id": scene_id,
            "duration": scene["duration"],
            "visual_prompt": visual,
            "action": action,
            "camera": camera,
            "dialogue": scene.get("dialogue", ""),
            "voiceover": scene.get("voiceover", ""),
            "music": scene.get("music", "auto"),
            "sound_effects": scene.get("sound_effects", []),
        })

    return storyboard


def save_storyboard(storyboard: dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(storyboard, f, indent=2, ensure_ascii=False)