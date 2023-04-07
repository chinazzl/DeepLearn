class BasePy:
    country = 'china'

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def talk(self):
        print(self.name, " is talking Chinese")


if __name__ == "__main__":
    dy = BasePy("Henan", 25)
    dy.talk()
