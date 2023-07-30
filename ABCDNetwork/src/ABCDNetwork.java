import java.io.*;
import java.util.*;

/*
* This class constructs an A-B-C-D network, which is a three-layer network with a configurable number of input activations,
* hidden activations, and outputs. The network can either run or train. When running, it propagates forward and calculates
* the outputs from input and weight values. When training, it uses backpropagation and gradient descent to modify weights
* until the average error is below a set error threshold, or the number of iterations is above the set maximum.
*
* Table of Contents
* public static void main(String[] args) throws IOException
* public static void configNetwork(String fileName) throws IOException
* public static void allocateMemory()
* public static void populateParameters() throws IOException
* public static void setActivations(int inputIndex)
* public static void train()
* public static void run()
* public static void runForTrain()
* public static void checkTrainingCompletion()
* public static void randomizeWeights()
* public static void loadWeights() throws IOException
* public static void saveWeights() throws IOException
* public static double calculateError()
* public static double activationFunction(double n)
* public static double activationFunctionDeriv(double n)
* public static double randomDouble(double min, double max)
* public static void printConfig()
* public static void printTrainingExitInfo()
* public static void printTruthTable()
* public static void printReport()
*
* @author Victoria Han
* @date created March 29, 2022
*/
public class ABCDNetwork
{

   static int numLayers;                      // number of connectivity layers
   static int numInputs;                      // number of input activations
   static int numHiddens_k;                   // number of hidden activations in kth layer
   static int numHiddens_j;                   // number of hidden activations in jth layer
   static int numOutputs;                     // number of outputs
   static int numInputSets;                   // number of input sets (i.e. {0, 1})

   static int[] layerSize;                    // number of activations in each network layer
   static int[][] inputs;                     // input values used for running and training
   static int[][] truthTable;                 // expected output values for each set of input values
   static double[][][] weights;               // connection weights between activations
   static double[][] activations;             // all activation values (input, hidden, and output)

   static double[] calculatedOutput;          // output calculated from input and weights
   static double[] expectedOutput;            // expected output used for training and calculating error
   static double[] psi_k;                     // stores psi_k for different hiddens during training
   static double[] psi_j;                     // stores psi_j for different hiddens during training
   static double[] psi_i;                     // stores psi_i for different outputs during training
   static double[] theta_k;                   // stores theta_k for different hiddens during training
   static double[] theta_j;                   // stores theta_j for different hiddens during training
   static double[] theta_i;                   // stores theta_i for different outputs during training
   static double avgError;                    // average error for each training iteration
   static double trainingTime;                // time it takes to train in milliseconds

   static double minWeight;                   // minimum random weight
   static double maxWeight;                   // maximum random weight
   static double minError;                    // error threshold used to determine if training is complete
   static int maxIterations;                  // maximum number of iterations used to determine if training should stop
   static double learningFactor;              // determines amount weights should be modified during training
   static int numIterations;                  // counter for number of iterations passed during training

   static boolean isTraining;                 // determines whether network is training or running
   static boolean isTrainingDone;             // indicates if training is complete
   static boolean reachedMinError;            // indicates if average error passed below error threshold during training
   static boolean weightsLoaded;              // determines if weights are loaded from a file or randomly generated

   static String truthTableFileName;          // name of the file truth table will be read from
   static String weightsFileName;             // name of the file weights may be loaded from
   static String trainedWeightsFileName;      // name of the file modified weights will be loaded to after training

   /*
    * Sets up the network, trains or runs, then prints network configuration information and results from training
    * or running.
    *
    * @param args             stores command-line arguments
    * @throws IOException     if file input or output operations fail
    */
   public static void main(String[] args) throws IOException
   {
      String configFileName;
      if (args.length > 0)
      {
         configFileName = args[0];
      }
      else
      {
         configFileName = "config.txt";
      }

      configNetwork(configFileName);
      allocateMemory();
      populateParameters();

      if (isTraining)
      {
         train();
         saveWeights();
      }
      else
      {
         run();
      }

      printReport();
   } // public static void main(String[] args) throws IOException

   /*
    * Sets values for non-array variables.
    *
    * @param fileName         name of the file to be read
    * @throws IOException     if file input or output operations fail
    */
   public static void configNetwork(String fileName) throws IOException
   {
      Scanner sc = new Scanner(new FileReader(fileName));

      truthTableFileName = sc.nextLine();
      weightsFileName = sc.nextLine();
      trainedWeightsFileName = sc.nextLine();

      numLayers = sc.nextInt();
      numInputs = sc.nextInt();
      numHiddens_k = sc.nextInt();
      numHiddens_j = sc.nextInt();
      numOutputs = sc.nextInt();
      numInputSets = sc.nextInt();

      minWeight = sc.nextDouble();
      maxWeight = sc.nextDouble();
      maxIterations = sc.nextInt();
      minError = sc.nextDouble();
      learningFactor = sc.nextDouble();
      numIterations = 0;

      isTraining = sc.nextBoolean();
      isTrainingDone = false;
      reachedMinError = false;
      weightsLoaded = sc.nextBoolean();
   } // public static void configNetwork(String fileName) throws IOException

   /*
    * Allocates memory for network by initializing arrays.
    */
   public static void allocateMemory()
   {
      layerSize = new int[numLayers + 1];
      inputs = new int[numInputSets][numInputs];
      truthTable = new int[numOutputs][numInputSets];

      weights = new double[numLayers][][];
      weights[0] = new double[numInputs][numHiddens_k];
      weights[1] = new double[numHiddens_k][numHiddens_j];
      weights[2] = new double[numHiddens_j][numOutputs];

      activations = new double[numLayers + 1][];
      activations[0] = new double[numInputs];
      activations[1] = new double[numHiddens_k];
      activations[2] = new double[numHiddens_j];
      activations[3] = new double[numOutputs];

      calculatedOutput = new double[numOutputs];
      expectedOutput = new double[numOutputs];

      if (isTraining)
      {
         psi_k = new double[numHiddens_k];
         psi_j = new double[numHiddens_j];
         psi_i = new double[numOutputs];
         theta_k = new double[numHiddens_k];
         theta_j = new double[numHiddens_j];
         theta_i = new double[numOutputs];
      } // if (isTraining)
   } // public static void allocateMemory()

   /*
    * Populates layerSize, inputs, truthTable, and weights arrays with appropriate values.
    *
    * @throws IOException     if file input or output operations fail
    */
   public static void populateParameters() throws IOException
   {
      layerSize[0] = numInputs;
      layerSize[1] = numHiddens_k;
      layerSize[2] = numHiddens_j;
      layerSize[3] = numOutputs;

      Scanner sc = new Scanner(new FileReader(truthTableFileName));

      inputs[0][0] = sc.nextInt();             // 1st input set
      inputs[0][1] = sc.nextInt();

      inputs[1][0] = sc.nextInt();             // 2nd input set
      inputs[1][1] = sc.nextInt();

      inputs[2][0] = sc.nextInt();             // 3rd input set
      inputs[2][1] = sc.nextInt();

      inputs[3][0] = sc.nextInt();             // 4th input set
      inputs[3][1] = sc.nextInt();

      truthTable[0][0] = sc.nextInt();         // 1st output set
      truthTable[0][1] = sc.nextInt();
      truthTable[0][2] = sc.nextInt();
      truthTable[0][3] = sc.nextInt();

      truthTable[1][0] = sc.nextInt();         // 2nd output set
      truthTable[1][1] = sc.nextInt();
      truthTable[1][2] = sc.nextInt();
      truthTable[1][3] = sc.nextInt();

      truthTable[2][0] = sc.nextInt();         // 3rd output set
      truthTable[2][1] = sc.nextInt();
      truthTable[2][2] = sc.nextInt();
      truthTable[2][3] = sc.nextInt();

      if (weightsLoaded)
      {
        loadWeights();
      }
      else
      {
         randomizeWeights();
      }
   } // public static void populateParameters() throws IOException

   /*
    * Sets the input activation values in the activations array.
    *
    * @param inputIndex    indicates which set of input values from the inputs array to use
    */
   public static void setActivations(int inputIndex)
   {
      for (int i = 0; i < numInputs; i++)
      {
         activations[0][i] = inputs[inputIndex][i];
      }
   }

   /*
    * Trains the network by using backpropagation and gradient descent to minimize error. Training stops if average error
    * is below the error threshold, or the maximum number of iterations has been reached.
    */
   public static void train()
   {
      double totalError;

      double startTime = System.currentTimeMillis();

      while (!isTrainingDone)
      {
         totalError = 0.0;

         for (int inputIndex = 0; inputIndex < numInputSets; inputIndex++)
         {
            setActivations(inputIndex);
            for (int i = 0; i < numOutputs; i++)
            {
               expectedOutput[i] = truthTable[i][inputIndex];
            }
            runForTrain();

            for (int j = 0; j < numHiddens_j; j++)
            {
               double omega_j = 0.0;
               for (int i = 0; i < numOutputs; i++)
               {
                  weights[2][j][i] += learningFactor * activations[2][j] * psi_i[i];
                  omega_j += psi_i[i] * weights[2][j][i];
               }

               psi_j[j] = omega_j * activationFunctionDeriv(theta_j[j]);

            } // for (int j = 0; j < numHiddens; j++)

            for (int k = 0; k < numHiddens_k; k++)
            {
               double omega_k = 0.0;
               for (int j = 0; j < numHiddens_j; j++)
               {
                  weights[1][k][j] += learningFactor * activations[1][k] * psi_j[j];
                  omega_k += psi_j[j] * weights[1][k][j];
               }

               double psi_k = omega_k * activationFunctionDeriv(theta_k[k]);

               for (int m = 0; m < numInputs; m++)
               {
                  weights[0][m][k] += learningFactor * activations[0][m] * psi_k;
               }
            } // for (int k = 0; k < numHiddens_k; k++)

            run();
            totalError += calculateError();
            numIterations++;

         } // for (int inputIndex = 0; inputIndex < numInputSets; inputIndex++)

         avgError = totalError / numInputSets;
         checkTrainingCompletion();

      } // while (!isTrainingDone)

      trainingTime = System.currentTimeMillis() - startTime;

   } // public static void train()

   /*
    * Calculates outputs through forward propagation using input activations and weights. Populates activations
    * array with appropriate values.
    */
   public static void run()
   {
      for (int n = 0; n < numLayers; n++)                         // iterates through all activation layers
      {
         for (int j = 0; j < layerSize[n + 1]; j++)               // iterates through all activations in layer n + 1
         {
            double hiddenSum = 0.0;

            for (int k = 0; k < layerSize[n]; k++)                // iterates through all activations in layer n
            {
               hiddenSum += activations[n][k] * weights[n][k][j]; // computes hidden activations
            }
            activations[n + 1][j] = activationFunction(hiddenSum);
         } // for (int j = 0; j < layerSize[n + 1]; j++)
      } // for (int n = 0; n < numLayers; n++)

      for (int i = 0; i < numOutputs; i++)
      {
         calculatedOutput[i] = activations[numLayers][i];
      }
   } // public static void run()

   /*
    * Populates activations array with appropriate values through forward propagation using input activations and weights.
    * Calculates and stores values to be used during training.
    */
   public static void runForTrain()
   {
      for (int k = 0; k < numHiddens_k; k++)
      {
         theta_k[k] = 0.0;
         for (int m = 0; m < numInputs; m++)
         {
            theta_k[k] += activations[0][m] * weights[0][m][k];
         }
         activations[1][k] = activationFunction(theta_k[k]);
      } // for (int k = 0; k > numHiddens_k; k++)

      for (int j = 0; j < numHiddens_j; j++)
      {
         theta_j[j] = 0.0;
         for (int k = 0; k < numHiddens_k; k++)
         {
            theta_j[j] += activations[1][k] * weights[1][k][j];
         }
         activations[2][j] = activationFunction(theta_j[j]);
      } // for (int j = 0; j < numHiddens_j; j++)

      for (int i = 0; i < numOutputs; i++)
      {
         theta_i[i] = 0.0;
         for (int j = 0; j < numHiddens_j; j++)
         {
            theta_i[i] += activations[2][j] * weights[2][j][i];
         }
         activations[3][i] = activationFunction(theta_i[i]);

         double T_i = expectedOutput[i];
         double omega_i = T_i - activations[3][i];
         psi_i[i] = omega_i * activationFunctionDeriv(theta_i[i]);
      } // for (int i = 0; i < numOutputs; i++)
   } // public static void runForTrain()

   /*
    * Checks if training is complete. Training is complete if either average error is below the error threshold or
    * the max number of iterations has been reached.
    */
   public static void checkTrainingCompletion()
   {
      if (avgError <= minError)
      {
         reachedMinError = true;
         isTrainingDone = true;
      }
      else if (numIterations >= maxIterations)
      {
         isTrainingDone = true;
      }
   } // public static void checkTrainingCompletion()

   /*
    * Fills weights array with random values between minWeight and maxWeight.
    */
   public static void randomizeWeights()
   {
      for (int n = 0; n < numLayers; n++)                         // iterates through all activation layers
      {
         for (int k = 0; k < layerSize[n]; k++)                   // iterates through all activations in layer n
         {
            for (int j = 0; j < layerSize[n + 1]; j++)            // iterates through all activations in layer n + 1
            {
               weights[n][k][j] = randomDouble(minWeight, maxWeight);
            }
         }
      } // for (int n = 0; n < numLayers; n++)
   } // public static void randomizeWeights()

   /*
    * Loads weights values from an input file to the weights array.
    *
    * @throws IOException     if file input or output operations fail
    */
   public static void loadWeights() throws IOException
   {
      Scanner sc = new Scanner(new FileReader(weightsFileName));

      for (int n = 0; n < numLayers; n++)                         // iterates through all activation layers
      {
         for (int k = 0; k < layerSize[n]; k++)                   // iterates through all activations in layer n
         {
            for (int j = 0; j < layerSize[n + 1]; j++)            // iterates through all activation in layer n + 1
            {
               weights[n][k][j] = sc.nextDouble();
            }
         }
      } //  for (int n = 0; n < numLayers; n++)
   } // public static void loadWeights() throws IOException

   /*
    * Saves current values in weights array to an output file.
    *
    * @throws IOException     if file input or output operations fail
    */
   public static void saveWeights() throws IOException
   {
      BufferedWriter bw = new BufferedWriter(new FileWriter(trainedWeightsFileName));

      for (int n = 0; n < numLayers; n++)                         // iterates through all activation layers
      {
         for (int k = 0; k < layerSize[n]; k++)                   // iterates through all activations in layer n
         {
            for (int j = 0; j < layerSize[n + 1]; j++)            // iterates through all activations in layer n + 1
            {
               bw.write(weights[n][k][j] + " ");
            }
            bw.newLine();
         }
      } // for (int n = 0; n < numLayers; n++)

      bw.newLine();
      bw.write("Network configuration: " + numInputs + "-" + numHiddens_k + "-" + numHiddens_j + "-" + numOutputs);
      bw.close();
   } // public static void saveWeights() throws IOException

   /*
    * Calculates error based on calculated output and expected output values.
    *
    * @return     the calculated error
    */
   public static double calculateError()
   {
      double totalError = 0.0;

      for (int i = 0; i < numOutputs; i++)
      {
         double difference = expectedOutput[i] - calculatedOutput[i];
         totalError += 0.5 * difference * difference;
      }

      return totalError / numOutputs;
   } // public static double calculateError()

   /*
    * The activation function used in forward propagation to calculate hidden and output activations.
    *
    * @param n    input value for the activation function
    * @return     output value for the activation function with input n
    */
   public static double activationFunction(double n)
   {
      return 1.0 / (1.0 + Math.exp(-n));       // sigmoid function
   }

   /*
    * The derivative of activation function used in gradient descent during training.
    *
    * @param n    input value for the derivative of the activation function
    * @return     output value for the derivative of the activation function with input n
    */
   public static double activationFunctionDeriv(double n)
   {
      return activationFunction(n) * (1.0 - activationFunction(n));
   }

   /*
    * Returns a random double within the given range.
    *
    * @param min     the minimum random double that can be returned
    * @param max     the maximum random double that can be returned
    * @return        a random double in the range (min, max)
    */
   public static double randomDouble(double min, double max)
   {
      return Math.random() * (max - min) + min;
   }

   /*
    * Prints the network configuration information.
    */
   public static void printConfig()
   {
      System.out.println("NETWORK CONFIGURATION");

      System.out.println("Number of inputs: " + numInputs);
      System.out.println("Number of hidden activations in kth layer: " + numHiddens_k);
      System.out.println("Number of hidden activations in jth layer: " + numHiddens_j);
      System.out.println("Number of outputs: " + numOutputs);
      System.out.println("Random weight range: (" + minWeight + ", " + maxWeight + ")");
      System.out.println("Max iterations: " + maxIterations);
      System.out.println("Error threshold: " + minError);
      System.out.println("Learning factor: " + learningFactor);

      System.out.println();
   } // public static void printConfig()

   /*
    * Prints training exit information, including reason for stopping training, average error, iterations reached, and
    * time elapsed.
    */
   public static void printTrainingExitInfo()
   {
      System.out.println("TRAINING EXIT REPORT");

      if (reachedMinError)
      {
         System.out.println("Training stopped because average error is below the error threshold.");
      }
      else
      {
         System.out.println("Training stopped because the maximum number of iterations was reached.");
      }
      System.out.println("Average error: " + avgError);
      System.out.println("Iterations reached: " + numIterations);
      System.out.println("Milliseconds elapsed during training: " + trainingTime);

      System.out.println();
   } // public static void printTrainingExitInfo()

   /*
    * Prints all sets of input values, expected output values, and output values calculated using current weights.
    */
   public static void printTruthTable()
   {
      System.out.println("TRUTH TABLE");

      for (int inputIndex = 0; inputIndex < numInputSets; inputIndex++)
      {
         setActivations(inputIndex);
         run();
         System.out.println("Input: (" + activations[0][0] + ", " + activations[0][1] + ")");

         for (int i = 0; i < numOutputs; i++)
         {
            System.out.println("Expected: " + truthTable[i][inputIndex]);
            System.out.println("Output: " + calculatedOutput[i]);
         }

         System.out.println();
      } // for (int inputIndex = 0; inputIndex < numInputSets; inputIndex++)
   } // public static void printTruthTable()

   /*
    * Prints report including network configuration information, training exit information, and a truth table
    * after training or running.
    */
   public static void printReport()
   {
      printConfig();

      if (isTraining)
      {
         printTrainingExitInfo();
      }

      printTruthTable();
   } // public static void printReport()

} // public class ABCDNetwork
