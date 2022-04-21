#include <Arduino.h>

/*#############################################################################
###############################################################################

                            START OF ML CONFIG

###############################################################################
#############################################################################*/

/******************************************************************
 * Network Configuration - customized per network 
 ******************************************************************/

const int PatternCount = 10;
const int InputNodes = 10;
const int HiddenNodes = 8;
const int OutputNodes = 1;
const float LearningRate = 0.5;
const float Momentum = 0.9;
const float InitialWeightMax = 0.5;
const float Success = 0.0004;
const int MaxTrainingCycle = 10000; // 2147483647

bool GoGoConnection = false;
bool model;

byte Input[PatternCount][InputNodes]; // var to hold training data --> gathered from GoGo Board

byte Target[PatternCount][OutputNodes]; // var to hold training data labels --> also from GoGo Board

/******************************************************************
 * End Network Configuration
 ******************************************************************/


int i, j, p, q, r;
int ReportEvery1000;
int RandomizedIndex[PatternCount];
long  TrainingCycle;
float Rando;
float Error;
float Accum;


float Hidden[HiddenNodes];
float Output[OutputNodes];
float HiddenWeights[InputNodes+1][HiddenNodes];
float OutputWeights[HiddenNodes+1][OutputNodes];
float HiddenDelta[HiddenNodes];
float OutputDelta[OutputNodes];
float ChangeHiddenWeights[InputNodes+1][HiddenNodes];
float ChangeOutputWeights[HiddenNodes+1][OutputNodes];


using namespace std;
using byte = unsigned char;

/*********************************************************************
 * Helper Methoids:
 *********************************************************************/

void toTerminal()
{

  for( p = 0 ; p < PatternCount ; p++ ) { 
    Serial.println(); 
    Serial.print ("  Training Pattern: ");
    Serial.println (p);      
    Serial.print ("  Input ");
    for( i = 0 ; i < InputNodes ; i++ ) {
      Serial.print (Input[p][i], DEC);
      Serial.print (" ");
    }
    Serial.print ("  Target ");
    for( i = 0 ; i < OutputNodes ; i++ ) {
      Serial.print (Target[p][i], DEC);
      Serial.print (" ");
    }
/******************************************************************
* Compute hidden layer activations
******************************************************************/

    for( i = 0 ; i < HiddenNodes ; i++ ) {    
      Accum = HiddenWeights[InputNodes][i] ;
      for( j = 0 ; j < InputNodes ; j++ ) {
        Accum += Input[p][j] * HiddenWeights[j][i] ;
      }
      Hidden[i] = 1.0/(1.0 + exp(-Accum)) ;
    }

/******************************************************************
* Compute output layer activations and calculate errors
******************************************************************/

    for( i = 0 ; i < OutputNodes ; i++ ) {    
      Accum = OutputWeights[HiddenNodes][i] ;
      for( j = 0 ; j < HiddenNodes ; j++ ) {
        Accum += Hidden[j] * OutputWeights[j][i] ;
      }
      Output[i] = 1.0/(1.0 + exp(-Accum)) ; 
    }
    Serial.print ("  Output ");
    for( i = 0 ; i < OutputNodes ; i++ ) {       
      Serial.print (Output[i], 5);
      Serial.print (" ");
    }
  }
}

byte * convertIntToByteArray(int predInt){
    static byte ArrayFromInt[InputNodes];

    for (byte i=0; i<InputNodes; i++) {
            byte state = bitRead(predInt, i);
            ArrayFromInt[i] = state;
            
        }

    return ArrayFromInt;
}
/*#############################################################################
###############################################################################

                            START OF ML METHODS

###############################################################################
#############################################################################*/


/**********************************************************************
 * ML Methods
 **********************************************************************/

void initModel (){

    /******************************************************************
    * Initialize HiddenWeights and ChangeHiddenWeights 
    ******************************************************************/

    for( i = 0 ; i < HiddenNodes ; i++ ) {    
        for( j = 0 ; j <= InputNodes ; j++ ) { 
        ChangeHiddenWeights[j][i] = 0.0 ;
        Rando = float(random(100))/100;
        HiddenWeights[j][i] = 2.0 * ( Rando - 0.5 ) * InitialWeightMax ;
        }
    }
    /******************************************************************
    * Initialize OutputWeights and ChangeOutputWeights
    ******************************************************************/

    for( i = 0 ; i < OutputNodes ; i ++ ) {    
        for( j = 0 ; j <= HiddenNodes ; j++ ) {
        ChangeOutputWeights[j][i] = 0.0 ;  
        Rando = float(random(100))/100;        
        OutputWeights[j][i] = 2.0 * ( Rando - 0.5 ) * InitialWeightMax ;
        }
    }
    Serial.println("Initial/Untrained Outputs: ");
    toTerminal();
}

void trainModel(){
    /******************************************************************
    * Begin training 
    ******************************************************************/
                                        
    for( TrainingCycle = 1 ; TrainingCycle <= MaxTrainingCycle ; TrainingCycle++) {    

    /******************************************************************
    * Randomize order of training patterns
    ******************************************************************/

        for( p = 0 ; p < PatternCount ; p++) {
        q = random(PatternCount);
        r = RandomizedIndex[p] ; 
        RandomizedIndex[p] = RandomizedIndex[q] ; 
        RandomizedIndex[q] = r ;
        }
        Error = 0.0 ;
    /******************************************************************
    * Cycle through each training pattern in the randomized order
    ******************************************************************/
        for( q = 0 ; q < PatternCount ; q++ ) {    
        p = RandomizedIndex[q];

    /******************************************************************
    * Compute hidden layer activations
    ******************************************************************/

        for( i = 0 ; i < HiddenNodes ; i++ ) {    
            Accum = HiddenWeights[InputNodes][i] ;
            for( j = 0 ; j < InputNodes ; j++ ) {
            Accum += Input[p][j] * HiddenWeights[j][i] ;
            }
            Hidden[i] = 1.0/(1.0 + exp(-Accum)) ;
        }

    /******************************************************************
    * Compute output layer activations and calculate errors
    ******************************************************************/

        for( i = 0 ; i < OutputNodes ; i++ ) {    
            Accum = OutputWeights[HiddenNodes][i] ;
            for( j = 0 ; j < HiddenNodes ; j++ ) {
            Accum += Hidden[j] * OutputWeights[j][i] ;
            }
            Output[i] = 1.0/(1.0 + exp(-Accum)) ;   
            OutputDelta[i] = (Target[p][i] - Output[i]) * Output[i] * (1.0 - Output[i]) ;   
            Error += 0.5 * (Target[p][i] - Output[i]) * (Target[p][i] - Output[i]) ;
        }

    /******************************************************************
    * Backpropagate errors to hidden layer
    ******************************************************************/

        for( i = 0 ; i < HiddenNodes ; i++ ) {    
            Accum = 0.0 ;
            for( j = 0 ; j < OutputNodes ; j++ ) {
            Accum += OutputWeights[i][j] * OutputDelta[j] ;
            }
            HiddenDelta[i] = Accum * Hidden[i] * (1.0 - Hidden[i]) ;
        }


    /******************************************************************
    * Update Inner-->Hidden Weights
    ******************************************************************/


        for( i = 0 ; i < HiddenNodes ; i++ ) {     
            ChangeHiddenWeights[InputNodes][i] = LearningRate * HiddenDelta[i] + Momentum * ChangeHiddenWeights[InputNodes][i] ;
            HiddenWeights[InputNodes][i] += ChangeHiddenWeights[InputNodes][i] ;
            for( j = 0 ; j < InputNodes ; j++ ) { 
            ChangeHiddenWeights[j][i] = LearningRate * Input[p][j] * HiddenDelta[i] + Momentum * ChangeHiddenWeights[j][i];
            HiddenWeights[j][i] += ChangeHiddenWeights[j][i] ;
            }
        }

    /******************************************************************
    * Update Hidden-->Output Weights
    ******************************************************************/

        for( i = 0 ; i < OutputNodes ; i ++ ) {    
            ChangeOutputWeights[HiddenNodes][i] = LearningRate * OutputDelta[i] + Momentum * ChangeOutputWeights[HiddenNodes][i] ;
            OutputWeights[HiddenNodes][i] += ChangeOutputWeights[HiddenNodes][i] ;
            for( j = 0 ; j < HiddenNodes ; j++ ) {
            ChangeOutputWeights[j][i] = LearningRate * Hidden[j] * OutputDelta[i] + Momentum * ChangeOutputWeights[j][i] ;
            OutputWeights[j][i] += ChangeOutputWeights[j][i] ;
            }
        }
        }

    /******************************************************************
    * Every 1000 cycles send data to terminal for display
    ******************************************************************/
        ReportEvery1000 = ReportEvery1000 - 1;
        if (ReportEvery1000 == 0)
        {
        Serial.println(); 
        Serial.println(); 
        Serial.print ("TrainingCycle: ");
        Serial.print (TrainingCycle);
        Serial.print ("  Error = ");
        Serial.println (Error, 5);

        toTerminal();

        if (TrainingCycle==1)
        {
            ReportEvery1000 = 999;
        }
        else
        {
            ReportEvery1000 = 1000;
        }
        }    


    /******************************************************************
    * If error rate is less than pre-determined threshold then end
    ******************************************************************/

        if( Error < Success ) {
            break ;  
        }
    }
}

/*********************************************************************
* Make a new prediction
*********************************************************************/

void predictNew(int predInt){
    Serial.println();
    Serial.println("NOW PREDICTING SOMETHING");
        
        byte NumArray[InputNodes];

        for (byte i=0; i<InputNodes; i++) {
            byte state = bitRead(predInt, i);
            NumArray[i] = state;
            Serial.print(state);
        }
        Serial.println();

        Serial.print("{");
        for( i = 0 ; i < InputNodes ; i++ ) {       
        Serial.print(NumArray[i]);
            if (i < InputNodes-1){
                Serial.print (", ");
            }
        }
        Serial.println("}");

    // byte test[InputNodes] = { 1, 1, 1, 0, 0, 1, 1 }; // Expected output (1,0,0,1)

    /******************************************************************
    * Compute hidden layer activations
    ******************************************************************/

            for( i = 0 ; i < HiddenNodes ; i++ ) {    
                Accum = HiddenWeights[InputNodes][i] ;
                for( j = 0 ; j < InputNodes ; j++ ) {
                    Accum += NumArray[j] * HiddenWeights[j][i] ;
                }
                Hidden[i] = 1.0/(1.0 + exp(-Accum)) ;
            }
    /******************************************************************
    * Compute output layer activations and calculate errors
    ******************************************************************/

        for( i = 0 ; i < OutputNodes ; i++ ) {    
            Accum = OutputWeights[HiddenNodes][i] ;
            for( j = 0 ; j < HiddenNodes ; j++ ) {
            Accum += Hidden[j] * OutputWeights[j][i] ;
            }
            Output[i] = 1.0/(1.0 + exp(-Accum)) ;   

            Serial.print("Output ");
            Serial.print(i);
            Serial.print(": ");
            Serial.println(Output[i]);
        }
        return;
}
