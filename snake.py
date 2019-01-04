from random import randint
import pygame
import numpy as np
from dna import DNA
from scipy.spatial import distance
import math

black = (0,0,0)
block_size = 10

class Snake:
    def __init__(self,gameDisplay,w=40,h=40,snake_len = 3,life =10000, food_energy = 10, dna = None, color = None):
        self.w = w
        self.h = h
        self.body = []
        self.xdir =1
        self.ydir = 0
        self.dir = 1
        self.grow = False
        self.snake_len = snake_len
        self.gameDisplay = gameDisplay
        self.create_snake()
        self.died = False
        if color is None:
            self.color = (randint(0,255),randint(0,255),randint(0,200))
        else:
            self.color = (randint(0,255),randint(0,255),randint(0,200))
        self.score = 0
        self.life = life
        self.life_orig = life
        self.food_energy = food_energy
        if dna is None:
            self.dna = DNA(6,4,[0,10])
        else:
            self.dna = dna
        self.food = []
        self.cur_state = None
        self.reward = 0

    def mix_color(self, color):
        new_color = (int((self.color[0]+color[0])/2),
                    int((self.color[0]+color[0])/2),
                    int((self.color[0]+color[0])/2))
        return new_color
    
    def get_fitness(self):
        return round(self.reward*0.2 + self.score*0.8,2)
    
    def create_snake(self):
        vertical = randint(0,1)
        random_x = randint(0,self.w)
        random_y = randint(0,self.h)
        
        random_x = int(self.w/2)
        random_y = int(self.h/2)
        
        if vertical:
            for i in range(0,self.snake_len):
                self.body.append([random_x, random_y+i])
        else:
            for i in range(0,self.snake_len):
                self.body.append([random_x+i, random_y])
                
    def show(self):
        pygame.draw.line(self.gameDisplay, self.color, [self.body[-1][0]*block_size, self.body[-1][1]*block_size],[self.food[0]*block_size, self.food[1]*block_size])
        for i,point in enumerate(self.body):
            pygame.draw.rect(self.gameDisplay, self.color, [point[0]*block_size, point[1]*block_size, block_size, block_size])

    def update_Q_table(self):
        if self.direction_blocked(0):
            self.dna.penalise_action(self.cur_state, 0, 5)
        elif self.direction_has_food(0):
            self.dna.reward_action(self.cur_state, 0, 5)
        if self.direction_blocked(1):
            self.dna.penalise_action(self.cur_state, 1, 5)
        elif self.direction_has_food(1):
            self.dna.reward_action(self.cur_state, 1, 5)
        if self.direction_blocked(2):
            self.dna.penalise_action(self.cur_state, 2, 5)
        elif self.direction_has_food(2):
            self.dna.reward_action(self.cur_state, 2, 5)
        if self.direction_blocked(3):
            self.dna.penalise_action(self.cur_state, 3, 5)
        elif self.direction_has_food(3):
            self.dna.reward_action(self.cur_state, 3, 5)
    

    def get_current_state(self):
        head = self.body[-1]
        xloc = self.food[0] - head[0]
        yloc = self.food[1] - head[1]
        
        return np.array([self.direction_blocked(0),
                         self.direction_blocked(1),
                         self.direction_blocked(2),
                         self.direction_blocked(3),
                         (xloc > 0),
                         (yloc > 0)
                         ]).tobytes()
    
    def get_angle_between_points(self):
        snake = self.body
        vector_a = np.array(self.food) - np.array(snake[-1])
        vector_b = np.array(self.body[-1]) - np.array(snake[-2])

        norm_of_vector_a = np.linalg.norm(vector_a)
        norm_of_vector_b = np.linalg.norm(vector_b)
        if norm_of_vector_a == 0:
            norm_of_vector_a = 10
        if norm_of_vector_b == 0:
            norm_of_vector_b = 10

        vector_a_normalized = vector_a / norm_of_vector_a
        vector_b_normalized = vector_b / norm_of_vector_b
        angle = math.atan2(
            vector_a_normalized[1] * vector_b_normalized[0] - vector_a_normalized[
                0] * vector_b_normalized[1],
            vector_a_normalized[1] * vector_b_normalized[1] + vector_a_normalized[
                0] * vector_b_normalized[0]) / math.pi
        #print(angle)
        return angle
    
    def move(self,food):
        #should move left-straight-right
        self.food = food
        self.cur_state = self.get_current_state()
        self.update_Q_table()
        if randint(0,100) != 100:
            dir = np.argmax(self.dna.Q_table[self.cur_state])
            #print(self.dna.Q_table[self.cur_state],dir)
            while (self.dir == 0 and dir == 1) or (self.dir == 1 and dir == 0) or (self.dir == 2 and dir == 3) or (self.dir == 3 and dir == 2):
                self.dna.penalise_action(self.cur_state, dir, 5)
                dir = np.argmax(self.dna.Q_table[self.cur_state])
        else:
            temp = np.argpartition(self.dna.Q_table[self.cur_state], -2)[-2:]
            dir = temp[0]
            while (self.dir == 0 and dir == 1) or (self.dir == 1 and dir == 0) or (self.dir == 2 and dir == 3) or (self.dir == 3 and dir == 2):
                self.dna.penalise_action(self.cur_state, dir, 5)
                temp = np.argpartition(self.dna.Q_table[self.cur_state], -2)[-2:]
                dir = temp[0]
            '''
            dir = randint(0,3)
            angle = self.get_angle_between_points()
            if angle < 0.25 and angle >-0.25:
                dir = self.dir
            while (self.dir == 0 and dir == 1) or (self.dir == 1 and dir == 0) or (self.dir == 2 and dir == 3) or (self.dir == 3 and dir == 2):
                dir = randint(0,3)
            '''

        old_dist = distance.euclidean(self.body[-1], self.food)
        self.update(dir)
        new_dist = distance.euclidean(self.body[-1], self.food)

        if(old_dist > new_dist):
            self.reward += 1;
            self.dna.reward_action(self.cur_state, dir, 1)
        else:
            self.reward -= 2
            self.dna.penalise_action(self.cur_state, dir, 2)

        self.life -= 1

    def setDir(self,x,y):
        self.xdir = x
        self.ydir = y
        
    def update(self,dir):
        self.dir = dir
        if dir == 0:
                self.setDir(-1, 0);
        elif dir == 1:
                self.setDir(1, 0);
        if dir == 2:
                self.setDir(0, -1);
        elif dir == 3:
                self.setDir(0, 1);
        head = self.body[-1].copy()
        if(not self.grow):
            self.body.pop(0)
        else:
            self.life += self.life_orig/4
            self.dna.reward_action(self.cur_state, dir, 5)
            self.score+= 1
            self.reward += 1;
        self.grow = False
        head[0] += self.xdir
        head[1] +=self.ydir
        self.body.append(head)
        self.died = self.collision()
        if self.died and self.life > 0:
            #print(self.dna.Q_table[self.cur_state],dir)
            self.dna.penalise_action(self.cur_state, dir, 10)
            

    def direction_blocked(self,dir):
        if dir == 0:
            xdir = -1
            ydir = 0
        elif dir == 1:
            xdir = 1
            ydir = 0
        if dir == 2:
            xdir = 0
            ydir = -1
        elif dir == 3:
            xdir = 0
            ydir = 1
        snake = self.body.copy()
        head = snake[-1].copy()
        snake.pop(0)
        head[0] += xdir
        head[1] += ydir
        snake.append(head)
        if self.collision(snake):
            return True
        else:
            return False

    def direction_has_food(self,dir):
        if dir == 0:
            xdir = -1
            ydir = 0
        elif dir == 1:
            xdir = 1
            ydir = 0
        if dir == 2:
            xdir = 0
            ydir = -1
        elif dir == 3:
            xdir = 0
            ydir = 1
        snake = self.body.copy()
        head = snake[-1].copy()
        head[0] += xdir
        head[1] += ydir
        if head == self.food:
            return True
        else:
            return False

    def collision(self,snake = None):
        if snake is None:
            predict = False
            snake = self.body
        else:
            predict = True
        x = snake[-1][0]
        y = snake[-1][1]
        if(x > self.w-1 or x < 0 or y > self.h-1 or y < 0):
            #if not predict: print("Hit wall")
            return True
        elif self.life == 0:
            #if not predict: print("No Life")
            return True
        elif (snake[-1] in snake[0:-1]):
            #if not predict: print("Hit body")
            return True
        else:
            return False
