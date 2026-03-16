import cv2
import yaml

from utils import draw_detections, ensure_parent, load_model, run_inference


def main() -> None:
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    inference_config = config.get("inference") or {}
    image_config = inference_config.get("image") or {}

    image = cv2.imread(image_config["input"])
    if image is None:
        raise FileNotFoundError(f"Unable to read input image: {image_config['input']}")

    model = load_model(inference_config["weights"])
    predictions = run_inference(model, image, conf=inference_config.get("conf", 0.25))
    rendered = draw_detections(image, predictions[0])

    ensure_parent(image_config["output"])
    cv2.imwrite(image_config["output"], rendered)
    print(f"Saved prediction to {image_config['output']}")


if __name__ == "__main__":
    main()
