class Experiment:
    def __init__(self, name, real_data_proportions, generated_data_proportions):
        self.name = name
        self.real = real_data_proportions 
        self.generated = generated_data_proportions

    def show(self):
        print("Experiment: " + self.name)
        print("real:")
        print(self.real)
        print("generated:")
        print(self.generated)
        print()


real_data = {
    "gospel": 125,
    "country": 415,
    "rap": 500,
    "metal": 350
}

baseline = Experiment(
    "baseline",
    real_data, 
    {key: 0 for key, val in real_data.items()}
)

evened_out = Experiment(
    "evened_out",
    real_data,
    {key: 500 - val for key, val in real_data.items()}
)

two_x = Experiment(
    "two_x",
    real_data,
    {key: val for key, val in real_data.items()}
)

three_x = Experiment(
    "three_x",
    real_data,
    {key: val * 2 for key, val in real_data.items()}
)

four_x = Experiment(
    "four_x",
    real_data,
    {key: val * 3 for key, val in real_data.items()}
)

synthetic_uneven = Experiment(
    "synthetic_uneven",
    {key: 0 for key, val in real_data.items()},
    real_data
)

synthetic_even = Experiment(
    "synthetic_even",
    {key: 0 for key, val in real_data.items()},
    {key: 500 for key, val in real_data.items()}
)

EXPERIMENTS = [
    baseline,
    evened_out,
    two_x,
    three_x,
    four_x,
    synthetic_uneven,
    synthetic_even
]

for e in EXPERIMENTS:
    e.show()
    