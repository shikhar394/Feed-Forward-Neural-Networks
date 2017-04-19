# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 03:37:50 2017

@author: shikhar
"""

import math

class Matrix():

		def __init__(self, List, Name= ' '):
				self.List = List
				self.Name = Name
				try:
					self.num_columns = len(List[0])
					self.num_rows = len(List)
				except TypeError:
					self.num_columns = len(List)
					self.num_rows = 1

		def __str__(self):
				return ("Dimensions of Matrix {}: {} X {}\n".format(self.Name, self.num_rows, self.num_columns))

		def __getitem__(self, key):
				return self.List[key]

		def __setitem__(self, key, value):
				self.List[key] = value


		def __eq__(self, other):
				if isinstance(other, Matrix):
						return self.List == other.List
				return TypeError

		def __ne__(self, other):
				if isinstance(other, Matrix):
						return self_List != other.List
				return TypeError



		def print_matrix(self):
				"""
				Prints given matrix in readable format
				"""
				print(("Dimensions of Matrix {}: {} X {}\n".format(self.Name, self.num_rows, self.num_columns)))
				for i in range(self.num_rows):
					for j in range(self.num_columns):
						if self.num_rows > 1:
							print(self.List[i][j], end="  ")
						else:
							print(self.List[j], end = "  ")
					print('\n')



		def dot_product(self, self_List, other):
				"""
				Accepts 2 Lists and finds their dot products. Should be of the same size
				"""
				return sum(self_List[i]* other[i] for i in range(self.num_columns))

		def transpose(self):
				"""
				Swaps rows and columns of a given matrix
				"""
				temp_matrix = []

				for j in range(self.num_columns):
					temp_matrix.append([None for i in range(self.num_rows)])

				for i in range(self.num_rows):
					for j in range(self.num_columns):
						temp_matrix[j][i] = self.List[i][j]

				return Matrix(temp_matrix, "Transpose" + str(self.Name))


		def Theta_X(self, other):
				"""
				Matrix Multiplication. Finds dot product of 2 given matrices. 
				Theta_X product for sigmoid function required for activation.
				"""
				Matrix_Product = []
				for j in range(other.num_rows):
					Matrix_Product.append(self.dot_product(self.List, other.List[j]))

				Matrix_Product = Matrix(Matrix_Product, "MatrixProduct") #For matrices with 1 row
				return Matrix_Product


		def reset_rowcol(self):
				"""
				Resets the number of rows and columns according to their new values
				"""
				try:
					self.num_columns = len(self.List[0])
					self.num_rows = len(self.List)
				except TypeError:
					self.num_columns = len(self.List)
					self.num_rows = 1



		def add_bias(self):
				"""
				Adds bias to the input layer
				"""
				if self.num_rows > 1:
					for i in range(self.num_rows):
						self.List[i].insert(0, 1)

				else:
					self.List.insert(0,1)
				self.reset_rowcol()


		def sigmoid(self):
				"""
				Performs the sigmoid function on every element of a given list, mostly hidden and output layers
				"""
				for i in range(len(self.List)):
					self.List[i] = (1/ (1 + math.exp(-(self.List[i]))))


class Neural_Networks():


		def __init__(self, Input_File, Weight_Files,  Output_File):
				"""
				Accepts Input File, all the Weight Files (Which further tracks number of hidden layers)
				and Output File and converts them all into matrices using the class
				"""
				self.input_matrix = []
				self.input_layer_File = Input_File
				self.weights_Matrix = []
				self.weights_Files = Weight_Files #Can be multiple depending on the number of layers
				self.outputs_Matrix = []
				self.estimated_outputs = []
				self.outputs_File = Output_File


		def open_OutputFiles(self):
				"""
				Converts outputFile into a Matrix
				"""
				temp_row = []
				for row in open(self.outputs_File):
						self.outputs_Matrix.append([1 if (i == int(row)-1) else 0 for i in range(10)])
				#	self.outputs_Matrix.append(int(row)-1) #PreProcessing to set the outputs to desired range
				self.outputs_Matrix  =  Matrix(self.outputs_Matrix, "Output Matrix")


		def open_InputFiles(self):
				"""
				Opens the input files
				"""
				Input_File = open(self.input_layer_File)
				temp_matrix = []
				count = 0
				for row in Input_File:
						count += 1
						for data_point in row.strip().split(','):
								temp_matrix.append(float(data_point))
						self.input_matrix.append(temp_matrix)
						temp_matrix = []
				Input_File.close()
				self.input_matrix = Matrix(self.input_matrix, "Input Matrix")
	
		
		def open_WeightFiles(self):
				"""
				Opens the various weightFiles and stores them as seperate matrices in a list
				"""
				temp_row = []
				temp_matrix = []
				count = 1
				Weight_Files = [File for File in self.weights_Files]
				for File in Weight_Files:
						for row in open(File):
								for data_point in row.strip().split(','):
										temp_row.append(float(data_point))
								temp_matrix.append(temp_row)
								temp_row = []
						self.weights_Matrix.append(Matrix(temp_matrix, "Weights"+str(count)))
						temp_matrix = []
						count += 1

		def open_Files(self):
				self.open_InputFiles()
				self.open_WeightFiles()
				self.open_OutputFiles()


		def eval_nextLayer(self, activation_prevLayer, Weights):
				"""
				Takes an activated layer, uses it as an input layer and uses the repective weights to produce the next layer
				"""
				activation_prevLayer.add_bias()
				Next_Layer = []
				Next_Layer = activation_prevLayer.Theta_X(Weights)
				Next_Layer.sigmoid()
				return Next_Layer

				
		def Forward_Propagate(self):
				"""
				Takes all the input examples from a given input file and works through the different weights
				and hidden layers to produce the output
				"""
				self.open_Files()
				for i in range(self.input_matrix.num_rows): #All examples : 5000
						Input_Layer = Matrix(self.input_matrix[i], "Feature"+str(i+1)) #Converts a given feature into matrix
						count = 0
						for Weights in self.weights_Matrix: #All the given weight files that were converted into matrices while opening the file
								Input_Layer = self.eval_nextLayer(Input_Layer, Weights)
								count += 1
						self.estimated_outputs.append(Input_Layer.List) 
				self.estimated_outputs = Matrix(self.estimated_outputs)

		def Process_Outputs_Error(self):
				"""
				Checks all the outputs produced by the neural net against the given labels to produce the error rate
				"""

				Error = 0
				print(self.input_matrix.num_rows)
				for i in range(self.input_matrix.num_rows):
						if self.estimated_outputs_processing(self.estimated_outputs[i]) != self.outputs_Matrix[i]:
								Error += 1
				print("Number of Errors = ", Error)
				return (Error/self.input_matrix.num_rows)*100


		def estimated_outputs_processing(self, Feature):
				"""
				Checks the label a given Feature points to
				"""
				max_Feature = max(Feature)
				temp_Feature = []
				for i in range(len(Feature)):
						if Feature[i] == max_Feature:
								temp_Feature.append(1)
						else:
								temp_Feature.append(0)
				return temp_Feature

		def loss_Function(self):
				"""
				Calculates the loss function 
				"""
				summation = 0
				for data_point in range(self.input_matrix.num_rows):
						summation += self.cost_function(data_point)
				return summation/self.input_matrix.num_rows

		def cost_function(self, data_point):
				summation = 0
				for k in range(len(self.estimated_outputs[data_point])): #Number of classes
						summation += self.outputs_Matrix[data_point][k] * (math.log(self.estimated_outputs[data_point][k])) \
												 + (1-self.outputs_Matrix[data_point][k]) * (math.log(1 - self.estimated_outputs[data_point][k]))
				return (-summation)



if __name__ == "__main__":  
	 N = Neural_Networks("ps5_data.csv", ("ps5_theta1.csv", "ps5_theta2.csv"), "ps5_data-labels.csv")
	 N.Forward_Propagate()
	 print("Error Percentage = {} %".format(N.Process_Outputs_Error()))
	 print("Loss Value = {}".format(N.loss_Function()))

