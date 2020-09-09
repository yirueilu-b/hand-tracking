# Hand Tracking

Inference Google's MediaPipe hand tracking models with Python

## Usage

`pip install -r requirements.txt`

`jupyter notebook`

Run `explore_traking_workflow.ipynb` to see how these two models work.

Run `python palm_detector_test.py` to test only the palm detector with a web cam.

Run `python hand_tracking_test.py` to test hand tracking with a web cam.

( multiple hands tracking is available )

## Result

- `palm_detector_test.py`

    ![](https://i.imgur.com/6QWVJyTm.png)

- `hand_tracking_test.py`

    ![](https://i.imgur.com/08LCtzMm.png)
    
    ![](https://i.imgur.com/MUmtcujm.png)
    
    ![](https://i.imgur.com/kP3h1Gp.gif)
    

## TODO

- Optimize code to increase performance

- More visualization options

- Modify workflow as MediaPipe ( palm detection runs in some specific situations )

## Acknowledgments

This work is a study of models developed by Google and distributed as a part of the [Mediapipe](https://github.com/google/mediapipe) framework. Big thanks to [wolterlw/hand_tracking](https://github.com/wolterlw/hand_tracking), [metalwhale/hand_tracking](https://github.com/metalwhale/hand_tracking) and [JuliaPoo/MultiHand-Tracking](https://github.com/JuliaPoo/MultiHand-Tracking) for python implement.
