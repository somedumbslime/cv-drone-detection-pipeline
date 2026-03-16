import cv2
import yaml

from utils import draw_detections, ensure_parent, load_model, run_inference


def main() -> None:
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    inference_config = config.get("inference") or {}
    video_config = inference_config.get("video") or {}

    cap = cv2.VideoCapture(video_config["input"])
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open input video: {video_config['input']}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ensure_parent(video_config["output"])
    writer = cv2.VideoWriter(
        video_config["output"],
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    model = load_model(inference_config["weights"])
    conf = inference_config.get("conf", 0.25)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        pred = run_inference(model, frame, conf=conf)[0]
        out_frame = draw_detections(frame, pred)
        writer.write(out_frame)

    cap.release()
    writer.release()
    print(f"Saved prediction video to {video_config['output']}")


if __name__ == "__main__":
    main()
