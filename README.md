# Hand Tracking

Inference Google's MediaPipe hand tracking models with Python

## Usage

`pip install -r requirements.txt`

`jupyter notebook`

Run `explore_traking_workflow.ipynb` to see how these two models work.

Run `python palm_detector_test.py` to test only the palm detector with a web cam.

Run `python hand_tracking_test.py` to test hand tracking with a web cam.

( multiple hands tracking is available )

## TODO

- Optimize code to increase performance

- More visualization options

- Modify workflow as MediaPipe ( palm detection runs in some specific situations )

## Acknowledgments

This work is a study of models developed by Google and distributed as a part of the [Mediapipe](https://github.com/google/mediapipe) framework. Big thanks to [wolterlw/hand_tracking](https://github.com/wolterlw/hand_tracking), [metalwhale/hand_tracking](https://github.com/metalwhale/hand_tracking) and [JuliaPoo/MultiHand-Tracking](https://github.com/JuliaPoo/MultiHand-Tracking) for python implement.
