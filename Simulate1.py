import numpy as np
import numpy.linalg.linalg as LA
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.transforms import Affine2D
import matplotlib.patches as patches
from matplotlib.text import Annotation
import scipy.stats as stats
import math


class World(object):
    def find(condition):
        res, = np.nonzero(np.ravel(condition))
        return res

    def __init__(self):
    #define initial environment
        self.initial_pos = np.array([5.3,26.7]) #define initial position of the subject 0,0 means centre (x,y)
        self.current_direction = []
        self.current_direction.append(np.array([-0.35, -0.90])) #define initial direction when subject begins to move
        self.current_pos = np.array([0,0])
        self.v = 0.4
        self.a = 0
        self.dt = 0.1
        self.speedup = 5 #init controlled device
        self.bodyradius = 1
        self.body = patches.Circle(xy=self.initial_pos, radius = self.bodyradius, fc = 'r', ec = 'r') #define body of subject to be a circle and red
        self.nobstacles = 3 #the number of obstacles
        self.obstacles = [] #empty list of obstacles in the form of circles (NOT INTEGER)
        self.obstacleradius = 1 #obstacles radius or dimension
        self.obstacles_current_direction = []
        self.vobs = [(0.2),(0.2),(0.1)] #obstacle velocity

        for i in range(self.nobstacles): #to define obstacles position
            obstacleposition = [(0.7, 4.5), (-11.2, -4.2), (-13.1, 9.8)]
            obstacle = patches.Circle(xy = obstacleposition[i], radius = self.obstacleradius, fc = 'black', ec = 'black') #define body of obstacles to be a circle and black
            self.obstacles.append(obstacle) #A list is an object. If you append another list onto a list,
            #the first list will be a single object at the end of the destined list.

        for i in range(self.nobstacles): #obstacle movement initial direction
            direct = [(0.69, 0.73), (0.69, 0.73), (-0.43, 0.90)]
            self.obstacles_current_direction.append(np.array(direct[i]))

        self.end = False
        self.pause = False
        self.crash = False
        self.goal = False


    def getdistance(self, body): #calculate distance between main body and obstacles (in the form of integer)

        distancelist = [] #define empty list
        for obstacle in self.obstacles:
            distance = np.array(body.get_center()) - np.array(obstacle.get_center()) #to get the array list of main body centre coordinates SUBTRACTED by array list obstacle centre coordinates
            distance = distance ** 2
            distance = (distance[0] + distance[1]) ** 0.5 #to find the shortest (hypotenuse distance main between body and object)
            distancelist.append(distance)

        return np.array(distancelist) #returns the result to the called

    def getdirection(self, body): #for main body to continuously sense the direction of main body to obstacles (in the form of coordinates)

        directionlist = []
        for obstacle in self.obstacles:
            direction = np.array(obstacle.get_center()) - np.array(body.get_center()) #direction of main body to obstacle
            distance = direction ** 2
            distance = (distance[0] + distance[1]) ** 0.5
            # line 63-65 to compute pythagoras theorem
            direction = direction / distance #divided by distance to find the direction in a single unit (as one unit but stil in (x,y) form)
            directionlist.append(direction) #add the calculated direction to the end of directionlist

        return np.array(directionlist)

    def getproxemics(self, body):  #calculate amount of space a person feels necessary to set apart from obstacles/other humans

        proxemicslist = []  # define empty list
        for obstacle in self.obstacles:
            distancecoordinate = np.array(body.get_center()) - np.array(obstacle.get_center())
            distance = distancecoordinate ** 2
            proxemics = (distance[0] + distance[1]) ** 0.5
            if proxemics <= 4.5: #intimate
                value = 4 #higher proxemics means the obstacles are nearer
                proxemicslist.append(value)
            elif proxemics > 4.5 and proxemics <= 12: #personal
                value = 3
                proxemicslist.append(value)
            elif proxemics > 12 and proxemics <= 37: #social
                value = 2
                proxemicslist.append(value)
            else: #public
                value = 1
                proxemicslist.append(value)

        return np.array(proxemicslist)  # returns the result to the called

    def controller(self, body):
        # design your own controller here
        for i in range(self.nobstacles):
            obstacledirection = self.obstacles_current_direction[i]
            move = self.obstacles[i].get_center()
            move += self.vobs[i] * obstacledirection #velocity magnitude * vector direction
            self.obstacles[i].set_center(move)
            if abs(move[0]) > 20 or abs(move[1]) > 20:
                self.vobs[i] = 0

        direction = self.current_direction[-1] #Negative numbers mean that you count from the right instead of the left. So, list[-1] refers to the last element, list[-2] is the second-last, and so on.
        distancelist = self.getdistance(body)
        directionlist = self.getdirection(body)

        stimuluslist = []
        for i in range(self.nobstacles):
            proxemicslist = self.getproxemics(body)
            alpha = 1
            beta = 1
            intensity_stimulus = alpha * proxemicslist[i] * (math.exp(beta * self.vobs[i]))
            stimuluslist.append(intensity_stimulus)


        if max(stimuluslist) >= 3: #[distancelist.argmin()]
            if distancelist.min() < (self.bodyradius + self.obstacleradius):
                self.crash = True
                self.v = 0
                self.vobs[distancelist.argmin()] = 0
            for obstacle in self.obstacles: #code for agent to change direction to avoid collisions
                awaydistance = math.exp(max(stimuluslist)) * (2 * self.obstacleradius) #try to use stimulus function to vary this => math.exp is used to have faster reaction to avoid collision
                distancenew = np.array(obstacle.get_center()) - np.array(body.get_center())
                distancenew = distancenew ** 2
                distancenew = (distancenew[0] + distancenew[1]) ** 0.5
                safedistance = ((distancenew ** 2) + (awaydistance ** 2)) ** 0.5
                theta = math.atan(awaydistance / safedistance)
                xshort = safedistance * math.sin(theta)
                xshort = round(-xshort, 2)
                yshort = safedistance * math.cos(theta)
                yshort = round(-yshort, 2)
                changedirection = (np.array([xshort, yshort]))
                self.current_direction.append(changedirection)
                direction = self.current_direction[-1]
                self.v = 0.0075
        pos = np.array(body.get_center()) + self.v * direction
        if abs(pos[0]) > 20 or abs(pos[1]) > 30:
            self.v = 0 #main body stops moving
            self.end = True #program ends
            for i in range(self.nobstacles):
                self.vobs[i] = 0
        if pos[1] == 0:
            self.v = 0
            self.goal = True
            for i in range(self.nobstacles):
                self.vobs[i] = 0

        return pos
                #return position of the main body to pos array

    def animate(self):

        fig, ax1 = plt.subplots(1, 1)#, gridspec_kw={'height_ratios': [10, 1]})
        body = self.body

        def init():

            ax1.axis("equal")
            ax1.set_xlim([-40, 40])
            ax1.set_ylim([-40, 40])
            ax1.set_title("Click On Main Display To Pause / Unpause")
            #ax2.set_title("Click On Sensor Graph To Change Time")
            ax1.add_artist(body)
            plt.plot([-30, 30], [0, 0], color="green") #Horizontal Line Finish (Target)
            for i in range(len(self.obstacles)):
                ax1.add_artist(self.obstacles[i])

            return body


        def draw(n):

            if self.pause is not True:
                tr = self.controller(body)
                body.set_center(tr)
            if self.end is True:
                ax1.set_title("End", color = 'blue')
            if self.crash is True:
                ax1.set_title("Crashed!!!", color = 'r')
            if self.goal is True:
                ax1.set_title("Goal Achieved", color='green')
            return body


        def onclick(event):
            if event.button == 1:
                self.pause = True
                ax1.set_title("Pause, Release To Continue", color='blue')


        def offclick(event):
            if event.button == 1:
                self.pause = False
                ax1.set_title("Continue, Click On Main Display To Pause", color='blue')


        ani = FuncAnimation(fig, draw, init_func=init, frames=100, interval=1000 * self.dt / self.speedup, blit=False)#, save_count=len(poses))

        fig.canvas.mpl_connect('button_press_event', onclick)
        fig.canvas.mpl_connect('button_release_event', offclick)

        plt.show()


w = World().animate()