from random import randint
from numpy.random import choice
from snake import Snake

class breed:
    def __init__(self,gameDisplay,w,h,snake_len,life, food_energy,population, snakes, fitness):
        self.w = w
        self.h = h
        self.snake_len = snake_len
        self.gameDisplay = gameDisplay
        self.life = life
        self.food_energy = food_energy
        self.population = population
        self.snakes = snakes
        self.fitness = fitness
        #print(fitness)

    def get_random_snake(self):
        '''
        idx = randint(0,int(self.population/2))
        #idx = randint(0,4)
        return self.snakes[self.fitness[idx][1]]
        '''
        return choice(self.snakes, 1, p=self.fitness)[0]

    def generate_snakes(self):
        snakes=[]
        for i in range(0,self.population):
            parent1 = self.get_random_snake()
            parent2 = self.get_random_snake()
            child_dna = parent1.dna.cross_dna(parent2.dna)
            child_color = parent1.mix_color(parent2.color)
            snake = Snake(self.gameDisplay, self.w, self.h,
                          snake_len=self.snake_len,
                          life = self.life, food_energy = self.food_energy,
                          dna=child_dna, color = child_color)
            snakes.append(snake)
        return snakes
