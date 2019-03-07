# EyeTracker with Blink Detection

This is a research project which detects user's eyes using webcam; extracts the iris and tracks it to detect where the user is looking at the screen. Also using the blink detection, different actions can be set to be performed when blinked looking at specific part of the monitor.

The research paper based on this project was presented on 2019 International Conference on Robotics,Electrical and Signal Processing Techniques (ICREST).

## Abstract
Gaze-based interaction between human and computers has opened a potential domain of effortless supervision. The eye is one of the most important organs to perceive information from our surroundings and it can be the most conspicuous way to interact with computers. Therefore, in this research, a system model has been introduced to control the multimedia player with eye gaze using an economical webcam, as well as provides higher precision and robustness. The goal is to control the play, pause, forward and backward functions of a multimedia player. The click event was triggered by detecting the user’s eye blink.

*Full paper available at <a href="https://ieeexplore.ieee.org/document/8644103">https://ieeexplore.ieee.org/document/8644103</a>

## How to Run

First prepare your environment and check if you have all the required library packages installed and configured.
Dependencies for this project are:
 - numpy
 - scipy
 - opencv
 - dlib
 - imutils
 - pyautogui
 
Then, Download or Clone this repository. Use your favorite IDE to run the .py file.

> If for some reason your camera is not turning on try changing 'src = 1' to 'src = 0' in line 169.
>
> vs = VideoStream(src=1).start()
>
> To
>
> vs = VideoStream(src=0).start()


### Side Note
If you are doing any project similar to this, feel free to contact me. I would try to cooperate as much as possible. We have plans to take this project further and add new features as we progress. Hopefully in future it wouldn't remain a pseudo-project anymore, rather would become a complete application.
