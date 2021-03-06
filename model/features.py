
import gen_features


FEATURES = [
    fet for fet, _ in gen_features.make_smoothed_features()
    if ("alpha0.002" not in fet and
        "alpha0.004" not in fet and
        "alpha0.006" not in fet and
        "alpha0.008" not in fet and
        "alpha0.01" not in fet and
        "alpha0.2" not in fet and
        "alpha0.4" not in fet and
        "alpha0.6" not in fet and
        "alpha0.7" not in fet and
        "alpha0.8" not in fet and
        "alpha0.9" not in fet and
        "tt512" not in fet and
        "tt1024" not in fet and
        "mathis model" not in fet)]
