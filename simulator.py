import pygame
from random import randint
from snake import Snake
from scipy.spatial import distance
from breed import breed
import numpy as np
import os
import sys
import pickle

white = (255,255,255)
black = (0,0,0)
red = (255,0,0)
blue = (0,0,255)
green = (0,255,0)
block_size = 10
FPS = 60

SNAKE_LEN = 2
LIFE = 10000
FOOD_ENERGY = 50
RENDER_GENEREATION_FREQ = 100

class Simulator:
    def __init__(self,w=40,h=40,population = 10, food_amount = 3):
        self.w = w
        self.h = h
        self.population = population
        self.food_amount = food_amount
        self.all_died = False
        self.generation = 1
        
        #create board
        self.render_init()
        self.font = pygame.font.SysFont('Arial', 25)
        #create snakes
        self.snakes_init()

        #create food
        self.food_init()

        self.highest_score = 0

    def snakes_init(self):
        self.snakes=[]
        for i in range(0,self.population):
            snake = Snake(self.gameDisplay, self.w, self.h, snake_len=SNAKE_LEN, life = LIFE, food_energy = FOOD_ENERGY)
            self.snakes.append(snake)

    def render_multi_line(self,text, x, y, fsize):
        lines = text.splitlines()
        for i, l in enumerate(lines):
            self.gameDisplay.blit(self.font.render(l, True, black), (x, y + fsize*i))

    def update_progress(self):
        text = "Generation:" + str(self.generation)+"\nHigest score from generation:" + str(self.high_score)
        self.render_multi_line(text, 20,20,25)
        pygame.display.update()

    def get_new_generation(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        print("Generation:" + str(self.generation)+"\nHigest score from generation:" + str(self.high_score))
        self.update_progress()
        new_gen = breed(self.gameDisplay, self.w, self.h, snake_len=SNAKE_LEN, life = LIFE, food_energy = FOOD_ENERGY,
                        population = self.population, snakes = self.snakes, fitness = self.fitness_data)
        self.snakes = new_gen.generate_snakes()
        self.generation+=1

    def food_init(self):
        self.foods=[]
        for i in range(0,self.food_amount):
            self.foods.append([randint(0,self.w-1), randint(0,self.h-1)])
        

    def render_init(self):
        pygame.init()
        self.gameDisplay = pygame.display.set_mode(((self.w)*block_size,(self.h)*block_size))
        pygame.display.set_caption('Simulator')
        self.gameDisplay.fill(white)
        self.clock = pygame.time.Clock()

    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        self.gameDisplay.fill(white)
        for food in self.foods:
            pygame.draw.rect(self.gameDisplay, red, [food[0]*block_size, food[1]*block_size,block_size,block_size])
        for snake in self.snakes:
            if not snake.died:
                snake.show()
        pygame.display.update()


    def move_snake(self):
        self.active_snake = 0
        for snake in self.snakes:
            if not snake.died:
                self.active_snake +=1
                snake.move(self.find_nearest_food(snake.body[-1]))
                #print(snake.life,snake.score)
        if self.active_snake == 0:
            self.all_died = True

    def remove_dead(self):
        snakes = []
        for snake in self.snakes:
            if not snake.died:
                snakes.append(snake)
        self.snakes=snakes

    def eat(self):
        for snake in self.snakes:
            for food in self.foods:
                if snake.body[-1] == food:
                    snake.grow = True
                    self.foods.remove(food)
                    self.foods.append([randint(0,self.w-1), randint(0,self.h-1)])
                    #print(self.foods)

    def normalize(self,v):
        norm = np.linalg.norm(v)
        if norm != 0: 
           v= v /norm
        m = min(v)
        v =  [temp-m for temp in v]
        total = sum(v)
        if total == 0:
            v =  [1 / len(v) for temp in v]
        else:
            v =  [temp / total for temp in v]
        return v
    def save_dna(self,dna):
        file_name = ".\\saved_snakes\\dna_gen" + str(self.generation) + "_score_" + str(self.highest_score) + ".pkl"
        print("saving to file" + str(file_name))
        with open(file_name, 'wb') as f:
            pickle.dump(dna.Q_table, f, pickle.HIGHEST_PROTOCOL)
        
    def collect_data(self):
        self.fitness_data = []
        i = 0
        for snake in self.snakes:
            self.fitness_data.append(snake.get_fitness())
            i+=1
        #print(self.fitness_data)
        self.fitness_data = self.normalize(self.fitness_data)
        #print(self.fitness_data)
        self.high_score = 0
        i = 0
        high_idx = 0
        for snake in self.snakes:
            if snake.score > self.high_score:
                self.high_score = snake.score
                high_idx = i
            i+=1
        if self.highest_score < self.high_score:
            self.highest_score = self.high_score
            self.save_dna(self.snakes[high_idx].dna)
        

    def reset_game(self):
        self.gameDisplay.fill(white)
        self.all_died = False
        
    def find_nearest_food(self, head):
        idx = 0
        i = 0
        dist = distance.euclidean(head, self.foods[0])
        for food in self.foods:
            temp_dist = distance.euclidean(head, food)
            if temp_dist < dist:
                dist = temp_dist
                idx = i
            i+=1
        #print(idx,self.foods)
        return self.foods[idx]
    
    def game_loop(self):
        while True:
            #check if food eaten
            self.eat()
            
            #Play next move
            self.move_snake()
            if not (self.generation-1)%RENDER_GENEREATION_FREQ:
                self.render()
            else:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        quit()

            if self.all_died:
                self.collect_data()
                self.reset_game()
                self.get_new_generation()

            #remove dead snakes
            #self.remove_dead()

            if not (self.generation-1)%RENDER_GENEREATION_FREQ:
                self.clock.tick(self.active_snake+FPS)
        quit()

if __name__ == "__main__":
        game = Simulator(w=40,h=40,population = 100, food_amount = 2)
        game.game_loop()
