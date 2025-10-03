# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 00:40:27 2024

@author: Hiba.Abidelkarem
"""
import random

# Class representing a type of food
class Food:
    def __init__(self, name, calories, proteins, fats):
        self.name = name
        self.calories = calories
        self.proteins = proteins
        self.fats = fats
    
    def __str__(self):
        return f"{self.name} (Calories: {self.calories}, Proteins: {self.proteins}, Fats: {self.fats})"

# Base class for all animals
class Animal:
    def __init__(self, name, weight, age):
        self.name = name
        self.weight = weight
        self.age = age
        self.foods = []

    def __str__(self):
        return f"Name-{self.name}, Weight-{self.weight}kg, Age-{self.age}yr"
    
    def __iter__(self):
        return iter(self.foods)
    
    def feed(self, food):
        self.foods.append(food)
    
    # Overload the + operator to combine two animals
    def __add__(self, other):
        new_name = f"{self.name[0]}{other.name[0]}{random.randint(1, 10000)}"
        new_weight = (self.weight + other.weight) / 2
        return Animal(new_name, new_weight, 0)

# Derived class for birds
class Bird(Animal):
    def __init__(self, name, weight, age, wingspan, beak_size):
        super().__init__(name, weight, age)
        self.wingspan = wingspan
        self.beak_size = beak_size

    def __str__(self):
        return f"{super().__str__()}, Wingspan-{self.wingspan}cm, Beak size-{self.beak_size}cm"
    
    # Overload the + operator to combine two birds
    def __add__(self, other):
        new_name = f"{self.name[0]}{other.name[0]}{random.randint(1, 10000)}"
        new_weight = (self.weight + other.weight) / 2
        return Bird(new_name, new_weight, 0, self.wingspan, other.beak_size)

# Derived class for fish
class Fish(Animal):
    def __init__(self, name, weight, age, gills_number, fins_number):
        super().__init__(name, weight, age)
        self.gills_number = gills_number
        self.fins_number = fins_number

    def __str__(self):
        return f"{super().__str__()}, Gills Number-{self.gills_number}, Fins Number-{self.fins_number}"
    
    # Overload the + operator to combine two fish
    def __add__(self, other):
        new_name = f"{self.name[0]}{other.name[0]}{random.randint(1, 10000)}"
        new_weight = (self.weight + other.weight) / 2
        return Fish(new_name, new_weight, 0, self.gills_number, other.fins_number)

# Derived class for flying fish, combining properties of Fish and Bird
class FlyingFish(Fish, Bird):
    def __init__(self, name, weight, age, gills_number, fins_number, wingspan, beak_size):
        Fish.__init__(self, name, weight, age, gills_number, fins_number)
        Bird.__init__(self, name, weight, age, wingspan, beak_size)

    def __str__(self):
        return f"{super(Fish).__str__()}, Wingspan-{self.wingspan}cm, Beak size-{self.beak_size}cm"

# Class representing a zoo
class Zoo:
    def __init__(self):
        self.animals = []

    # Add a new animal to the zoo
    def add_animal(self, animal):
        if not any(a.name == animal.name for a in self.animals):
            self.animals.append(animal)
            print("Success")
        else:
            print("Animal with this name already exists.")

    # Print all animals in the zoo
    def print_animals(self):
        if not self.animals:
            print("No animals")
        else:
            for animal in self.animals:
                print(animal)

    # Find an animal by name
    def find_animal(self, name):
        for animal in self.animals:
            if animal.name == name:
                return animal
        return None

    # Pair two animals to create an offspring
    def pair_animals(self, name1, name2):
        animal1 = self.find_animal(name1)
        animal2 = self.find_animal(name2)
        
        if not animal1 or not animal2:
            print("Animal does not exist")
            return
        
        # If both animals are of the same type
        if type(animal1) == type(animal2):
            offspring = animal1 + animal2
        # If one is a fish and the other is a bird, create a FlyingFish
        elif (isinstance(animal1, Fish) and isinstance(animal2, Bird)) or (isinstance(animal1, Bird) and isinstance(animal2, Fish)):
            new_name = f"{animal1.name[0]}{animal2.name[0]}{random.randint(1, 10000)}"
            fish = animal1 if isinstance(animal1, Fish) else animal2
            bird = animal1 if isinstance(animal1, Bird) else animal2
            offspring = FlyingFish(new_name, (fish.weight + bird.weight) / 2, 0, fish.gills_number, fish.fins_number, bird.wingspan, bird.beak_size)
        else:
            print("Not from the same kind")
            return
        
        self.add_animal(offspring)

    # Feed all animals of a specific type
    def feed_animals(self, animal_type, food):
        for animal in self.animals:
            if isinstance(animal, animal_type):
                animal.feed(food)

    # Print the food history of a specific animal
    def print_animal_food(self, name):
        animal = self.find_animal(name)
        if not animal:
            print("Animal does not exist")
            return
        
        for food in animal:
            print(food)

# Main function to interact with the zoo
def main():
    # Create a new zoo instance
    zoo = Zoo()

    while True:
        # Display the main menu
        print("Please choose option [1-6]:")
        print("1 - Add new animal")
        print("2 - Print all animals")
        print("3 - Pair animals")
        print("4 - Feed animals")
        print("5 - Print food history of an animal")
        print("6 - Exit")
        
        # Get the user's choice
        choice = int(input().strip())

        if choice == 1:
            # Adding a new animal
            print("What type of animal do you want to add? (1-General, 2-Bird, 3-Fish, 4-Flying Fish)")
            animal_type = int(input().strip())
            name = input("Name: ").strip()
            weight = float(input("Weight: ").strip())
            age = int(input("Age: ").strip())

            # Create an animal instance based on the selected type
            if animal_type == 1:
                animal = Animal(name, weight, age)
            elif animal_type == 2:
                wingspan = float(input("Wingspan: ").strip())
                beak_size = float(input("Beak size: ").strip())
                animal = Bird(name, weight, age, wingspan, beak_size)
            elif animal_type == 3:
                gills_number = int(input("Gills number: ").strip())
                fins_number = int(input("Fins number: ").strip())
                animal = Fish(name, weight, age, gills_number, fins_number)
            elif animal_type == 4:
                gills_number = int(input("Gills number: ").strip())
                fins_number = int(input("Fins number: ").strip())
                wingspan = float(input("Wingspan: ").strip())
                beak_size = float(input("Beak size: ").strip())
                animal = FlyingFish(name, weight, age, gills_number, fins_number, wingspan, beak_size)
            else:
                print("Invalid animal type")
                continue

            # Add the new animal to the zoo
            zoo.add_animal(animal)

        elif choice == 2:
            # Print all animals in the zoo
            zoo.print_animals()

        elif choice == 3:
            # Pair two animals to create an offspring
            name1 = input("First animal name? ").strip()
            name2 = input("Second animal name? ").strip()
            zoo.pair_animals(name1, name2)

        elif choice == 4:
            # Feed animals of a specific type
            print("What kind of animal would you like to feed? (1-General, 2-Bird, 3-Fish, 4-Flying Fish)")
            animal_type_choice = int(input().strip())
            food_name = input("Food name?\n").strip()
            calories = int(input("Calories?\n").strip())
            proteins = int(input("Proteins?\n").strip())
            fats = int(input("Fats?\n").strip())

            # Create a new food instance
            food = Food(food_name, calories, proteins, fats)

            # Feed the animals based on the selected type
            if animal_type_choice == 1:
                zoo.feed_animals(Animal, food)
            elif animal_type_choice == 2:
                zoo.feed_animals(Bird, food)
            elif animal_type_choice == 3:
                zoo.feed_animals(Fish, food)
            elif animal_type_choice == 4:
                zoo.feed_animals(FlyingFish, food)
            else:
                print("Invalid animal type")

        elif choice == 5:
            # Print the food history of a specific animal
            name = input("Animal name: ").strip()
            zoo.print_animal_food(name)

        elif choice == 6:
            # Exit the program
            print("Goodbye!")
            break

        else:
            # Handle invalid choices
            print("Invalid choice")

# Run the main function when the script is executed
if __name__ == "__main__":
    main()
