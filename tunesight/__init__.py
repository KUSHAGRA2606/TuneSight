"""TuneSight: a jal tarang that sees and listens.

Camera  -> bowl geometry and water level (fill ratio h/H)
Mic     -> modal frequencies and damping of each strike (source-filter / LPC model)
Physics -> f(h) = f0 / sqrt(1 + beta (h/H)^4)  links the two, learned online from taps
Fusion  -> which identical-looking bowl was struck, consistency checks, vision-guided tuning to a raga
"""
__version__ = "1.0.0"
