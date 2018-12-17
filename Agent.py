import numpy as np
import Interaction
from Teacher import Teacher
from DataFiles import DataFiles

class Agent(object):

    alpha = 0.3#0.1 #0.7
    gamma = 0.9 #0.4
    epsilon = 0.1 #0.25

    def __init__(self, scenario, simulator):
        self.scenario = scenario
        self.simulator = simulator
        self.numberOfStates = self.scenario.getNumberOfStates()
        self.numberOfActions = self.scenario.getNumberOfActions()
        self.Q = np.random.uniform(0.0,0.01,(self.numberOfStates,self.numberOfActions))
        #self.Q = np.zeros((self.numberOfStates,self.numberOfActions))
    #end of __init__ method
        
    def saveQValues(self, filename):
        files = DataFiles()
        files.createFile(filename)
        for i in range(self.numberOfStates):
            files.addFloatToFile(filename, self.Q[i])

    def loadQValues(self, filename):
        files = DataFiles()
        self.Q = files.readFloatFile(filename)
        
    def selectTeachers(self, filename):
        files = DataFiles()
        steps = files.readFile(filename)
        iterations = len(steps[0])
        tries = len(steps)
        finalRate = 0.01 #Considering the last 1% of performed actions

        totalFinalSteps = np.zeros(tries)
        minStep = iterations - int(iterations*finalRate)
        for i in range(tries):
            acum = 0
            for j in range(minStep, iterations):
                acum += steps[i][j]
            #endfor
            totalFinalSteps[i] = acum
        #endfor
            
        totalFinalStepsTidy = np.argsort(totalFinalSteps)
        bestAgent = totalFinalStepsTidy[0]
        medianAgent = totalFinalStepsTidy[int(tries/2)]
        worstAgent = totalFinalStepsTidy[tries-1]

        return bestAgent, medianAgent, worstAgent

    def selectAction(self, state):
        if (np.random.rand() <= self.epsilon):
            action = np.random.randint(self.numberOfActions)
        else:
            action = np.argmax(self.Q[state,:])
        #endIf
        return action
    #end of selectAction method
    
    def selectActionWithAffordances(self, state):
        if (np.random.rand() <= self.epsilon):
            while True:
                action = np.random.randint(self.numberOfActions)
                if state == 45:
                    break
                if self.scenario.getTransition(state, action) != -1:
                    break
                #endIf
            #endWhile
        else:
            tidyQ = np.argsort(self.Q[state,:])
            pos = self.numberOfActions - 1
            action = tidyQ[pos]
            while self.scenario.getTransition(state, action) == -1:
                if state == 45:
                    break
                pos = pos - 1
                action = tidyQ[pos]
            #endwhile                    
        #endIf
        return action
    #end of selectActionWithAffordances method

    def getAdvice(self, teacherAgent, state, consistencyProbability):
        #print teacherAgent.Q[state,:]
        tidyQ = np.argsort(teacherAgent.Q[state][:])
        
        if (np.random.rand() < consistencyProbability):
            #good advice
            pos = self.numberOfActions - 1
            actionNew = tidyQ[pos]
            while self.scenario.getTransition(state,actionNew) == -1 and state != 45:
                pos -= 1
                actionNew = tidyQ[pos]
            #end of while
        else:
            #bad advice
            pos = 0
            actionNew = tidyQ[pos]
            while self.scenario.getTransition(state,actionNew) == -1 and state != 45:
                pos += 1
                actionNew = tidyQ[pos]
            #end of while
        #endif
        return actionNew
    #end of getAdvice method

    def train(self, iter, affordances, teacherAgent, feedbackProbability, consistencyProbability):
        contCatastrophic = 0
        contFinalReached = 0
        steps = np.zeros(iter)
        rewards = np.zeros(iter)
        
        for i in range(iter):
            contSteps = 0
            accReward = 0
            self.scenario.resetScenario()
            state = self.scenario.getState()
            #choose an action with epsilon-greedy action selection and affordances
            if affordances == 1:
                action = self.selectActionWithAffordances(state)
            else:
                action = self.selectAction(state)

            #expisode
            while True:
                #perform action
                self.scenario.executeAction(action)
                contSteps += 1

                #get reward
                reward = self.scenario.getReward()
                accReward += reward

                #catastrophic state
                if reward == -1:
                    contCatastrophic += 1
                    self.Q[state,action] = -0.1
                    #print('Catastrophe state in IRL?')
                    break

                stateNew = self.scenario.getState()
                #receives or not interactive advice
                if (np.random.rand() < feedbackProbability):
                    #get advice
                    actionNew = self.getAdvice(teacherAgent, stateNew, consistencyProbability)
                else:
                    #new action
                    if affordances == 1:
                        actionNew = self.selectActionWithAffordances(stateNew)
                    else:
                        actionNew = self.selectAction(stateNew)
                    #endif
                #endif

                # updating Q-values
                self.Q[state, action] += self.alpha * (reward + self.gamma * 
                                         self.Q[stateNew,actionNew] - 
                                         self.Q[state,action])

                if reward == 1:
                    contFinalReached += 1
                    steps[i] = contSteps
                    break
                
                state = stateNew
                action = actionNew
            #end of while
            rewards[i]=accReward
        #end of for
        return steps,rewards
    #end of train method
        
    def trainDemo(self, iter, affordances, interactionMode, showSimulator):
        #This method is the same than train but does not consider feedbackProbability, consistency, nor teacherAgent.
        #In this case is consider interactionMode to switch between GUI and speech (depth onwards)
        for i in range(iter):
            self.scenario.resetScenario()
            state = self.scenario.getState()
            #choose an action with epsilon-greedy action selection and affordances
            if affordances == 1:
                action = self.selectActionWithAffordances(state)
            else:
                action = self.selectAction(state) 

            #episode
            while True:
                #perform action
                self.scenario.executeAction(action)
                if showSimulator == 1:
                    print 'Current state: ', state
                    print 'Action to perform: ',action
                    print '--------------------------------------'
                    self.simulator.performAnAction(action)

                #get reward
                reward = self.scenario.getReward()

                #catastrophe state, in affordance versions is not used
                if reward == -1:
                    self.Q[state,action] = -0.1
                    print 'Catastrophe state'
                    break
                
                stateNew = self.scenario.getState()

                #new action
                if affordances == 1:
                    actionNew = self.selectActionWithAffordances(stateNew)
                else:
                    actionNew = self.selectAction(stateNew)
                #endif

                if interactionMode == 1:
                    if Interaction.buttonPush != -1:
                        print 'Change the action, buttom pressed with value = ', Interaction.buttonPush
                        if self.scenario.getTransition(stateNew, Interaction.buttonPush) != -1:
                            actionNew = Interaction.buttonPush
                        else:
                            print 'Not possible to perform this action in this state'
                        #endIf
                        Interaction.buttonPush = -1
                    #endIf
 
                # updating Q-values
                self.Q[state, action] += self.alpha * (reward + self.gamma * 
                                         self.Q[stateNew,actionNew] - 
                                         self.Q[state,action])

                if reward == 1:
                    if showSimulator == 1 and i != iter-1:
                        print 'Restarting scenario!... '
                        self.simulator.restartScenario()
                        
                    break
                #endIf
                
                state = stateNew
                action = actionNew
            #end of while
        #end of for
    #end of trainDemo method

    def demoMovement(self, affordances, interactionMode):
        print "Inside the method demoMovement"
        # None
        if interactionMode == 0:
            state = self.scenario.getState()
            if affordances == 1:
                action = self.selectActionWithAffordances(state)
            else:
                action = self.selectAction(state) 
            while True:
                #select action
                
                Interaction.buttonPush = np.random.randint(self.numberOfActions)
                stateNew = self.scenario.getState()
                #new action
                if affordances == 1:
                    actionNew = self.selectActionWithAffordances(stateNew)
                else:
                    actionNew = self.selectAction(stateNew)
                #endif
                #perform actions
                if Interaction.buttonPush != -1:
                    print 'Change the action, buttom pressed with value = ', Interaction.buttonPush
                    if self.scenario.getTransition(stateNew, Interaction.buttonPush) != -1:
                        self.simulator.performAnAction(Interaction.buttonPush)
                    
                    else:
                            print 'Not possible to perform this action in this state' 
                    Interaction.buttonPush = -1
                #endIf
                state = stateNew
        #endIf
                action = actionNew
        #Using GUI
        if interactionMode == 1:
            Interaction.buttonPush = np.random.randint(self.numberOfActions)
            state = self.scenario.getState()
            if affordances == 1:
                action = self.selectActionWithAffordances(state)
            else:
                action = self.selectAction(state) 
            while True:
                stateNew = self.scenario.getState()
                if affordances == 1:
                    actionNew = self.selectActionWithAffordances(stateNew)
                else:
                    actionNew = self.selectAction(stateNew)
                #perform actions
                if Interaction.buttonPush != -1:
                    print 'Change the action, buttom pressed with value = ', Interaction.buttonPush
                    if self.scenario.getTransition(stateNew, Interaction.buttonPush) != -1:
                        self.simulator.performAnAction(Interaction.buttonPush)
                    else:
                        print 'Not possible to perform this action in this state'
                        Interaction.buttonPush = -1
                #endIf
                state = stateNew
        #endIf
        action = actionNew
        
        #Using DOCKS
        if interactionMode == 2:
            teacher = Teacher()
            while True:
                #perform actions
                if Interaction.sentence != "":
                    print 'Change the action, speech direction = ', Interaction.sentence
                    actionCode = teacher.actionToCode(Interaction.sentence)
                    self.simulator.performAnAction(actionCode)
                    Interaction.sentence = ""
                #endIf
        #endIf
            
        
    #end of demoMovement method

#end of class Agent
