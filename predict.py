from mlp import Model

if __name__ == "__main__":
    model = Model.load("networks/mymodel(1).mlp")
    print(model.x_max)
