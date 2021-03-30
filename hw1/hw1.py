import random
import string
import pdb


random.seed(0)


def random_name(length=5):
    names = list()
    for j in range(3):
        name = list()
        for i in range(10):  # an array of 10 random strings
            name.append(''.join(random.choice(string.ascii_lowercase) for l in range(5)))
        names.append(name)
    return names


class People:
    def __init__(self, first_names, middle_names, last_names, name_order):
        self.first_names = first_names
        self.middle_names = middle_names
        self.last_names = last_names
        self.name_order = name_order
        self.index = -1

    def __iter__(self):
        return self

    def __next__(self):
        self.index += 1
        if self.index > len(self.first_names):
            raise IndexError('name index out of range')
        elif self.name_order == 'first_name_first':
            return self.first_names[self.index] + ' ' + self.middle_names[self.index] + ' ' + self.last_names[
                self.index]
        elif self.name_order == 'last_name_first':
            return self.last_names[self.index] + ' ' + self.first_names[self.index] + ' ' + self.middle_names[
                self.index]
        elif self.name_order == 'last_name_with_comma_first':
            return self.last_names[self.index] + ', ' + self.first_names[self.index] + ' ' + self.middle_names[
                self.index]
        else:
            raise StopIteration

    def __call__(self, *args, **kwargs):
        # print('\nsorted list of just the last names:')
        sorted_names = sorted(self.last_names)
        for name in sorted_names:
            print(name)


def random_wealth(length=10):
    return [random.randint(0, 1000) for _ in range(length)]


class PeopleWithMoney(People):
    def __init__(self, first_names, middle_names, last_names, name_order, wealth):
        super().__init__(first_names, middle_names, last_names, name_order)
        self.wealth = wealth
        # self.index = -1

    def __next__(self):
        return People.__next__(self) + " " + str(self.wealth[self.index])

    def __call__(self, *args, **kwargs):
        # print('\nsorted list by wealth:')
        sorted_PeopleWithMoney = sorted(zip(self.first_names, self.middle_names, self.last_names, self.wealth),
                                        key=lambda people: people[-1])
        for person in sorted_PeopleWithMoney:
            print(str(person)[1:-1].replace(',', ' ').replace('\'', ''))


name_list = random_name()
people_1 = People(name_list[0], name_list[1], name_list[2], 'first_name_first')
people_2 = People(name_list[0], name_list[1], name_list[2], 'last_name_first')
people_3 = People(name_list[0], name_list[1], name_list[2], 'last_name_with_comma_first')

# Q 4&5: Iterating through the data stored in the People instance
# print('\nIterating through the data stored in the people_1: (first_name_first)')
iter1 = iter(people_1)
for i in range(10):
    print(next(iter1))

# print('\nIterating through the data stored in the people_2: (last_name_first)')
print("")
iter2 = iter(people_2)
for i in range(10):
    print(next(iter2))

# print('\nIterating through the data stored in the people_3: (last_name_with_comma_first)')
print("")
iter3 = iter(people_3)
for i in range(10):
    print(next(iter3))

# Q 6: apply the function-call operator `()' to an instance, it should print out a sorted
# list of just the last names
print("")
people_1()

wealth_list = random_wealth()
people_money = PeopleWithMoney(name_list[0], name_list[1], name_list[2], 'first_name_first', wealth_list)

# Q7: iterate through an instance of PeopleWithMoney
# print('\nIterating through the data stored in the people_money:')
print("")
iter4 = iter(people_money)
for i in range(10):
    print(next(iter4))

# Q7: make the instances of the subclass callable
print("")
people_money()

pdb.set_trace()
