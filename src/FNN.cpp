//Author: Ralph Heymsfeld
//28/06/2018

#include <math.h>
#include <Arduino.h>
#include <GoGoBoardArduino.h>
#include <stdexcept>
#include <ml.h>

using namespace std;
using byte = unsigned char;

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

/*
const byte Input[PatternCount][InputNodes] = {
  { 1, 1, 1, 1, 1, 1, 0 },  // 0
  { 0, 0, 1, 1, 0, 0, 0 },  // 1
  { 1, 1, 0, 1, 1, 0, 1 },  // 2
  { 1, 1, 1, 1, 0, 0, 1 },  // 3
  { 0, 1, 1, 0, 0, 1, 1 },  // 4
  { 1, 0, 1, 1, 0, 1, 1 },  // 5
  { 0, 0, 1, 1, 1, 1, 1 },  // 6
  { 1, 1, 1, 0, 0, 0, 0 },  // 7 
  { 1, 1, 1, 1, 1, 1, 1 },  // 8
  { 1, 1, 1, 0, 0, 1, 1 }   // 9
}; 
*/

byte Input[PatternCount][InputNodes];

byte Target[PatternCount][OutputNodes];

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
/*
    Serial.println();

    Serial.print("{");
    for( i = 0 ; i < InputNodes ; i++ ) {       
    Serial.print(ArrayFromInt[i]);
        if (i < InputNodes-1){
            Serial.print (", ");
        }
    }
    Serial.println("}");
*/
    return ArrayFromInt;
}

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

/***********************************************************************
 * GOGO BOARD METHODS
***********************************************************************/
bool dataStreamOpen (){
    bool receivingData = false;

    if (GoGoBoard.isGmessageAvailable("data-open")){
        
        String DataOpen = GoGoBoard.Gmessage("data-open");     
        receivingData = true;
        
    }else if (GoGoBoard.isGmessageAvailable("data-closed")){
        
        String DataClosed = GoGoBoard.Gmessage("data-closed");
        receivingData = false;
    }

    return receivingData;
}

void getGoGoConnection(){
    while(!GoGoConnection){
        Serial.println("NO CONNECTION TO GOGO BOARD");
        GoGoConnection = dataStreamOpen();
        delay(500);
    }
    Serial.println("CONNECTED TO GOGO BOARD");
}

int getDataToPredictFromGoGo(){

    int dataToPredict = 9999;
    bool dataReceived = false;

    Serial.println();
    Serial.println("waiting for data for prediction from GoGo Board...");

    if(GoGoConnection){

        GoGoBoard.sendGmessage("receive-data-to-predict", GoGoBoard.Gmessage("receive-data-to-predict"));

        while(!dataReceived){
            if (GoGoBoard.isGmessageAvailable("data-to-predict")){
                dataToPredict = GoGoBoard.Gmessage("data-to-predict").toInt();
                Serial.print("Received: ");
                Serial.println(dataToPredict);
                dataReceived = true;
            } 
        }    
    }else{
        Serial.println("NO CONNECTION TO GOGO BOARD");
    }
    
    if(dataToPredict != 9999){
            return dataToPredict;
    }else{
        String invalidData = "Invalid data received";
        Serial.println(invalidData);
        throw std::invalid_argument("Invalid data received");
    }
}

void getTrainingDataFromGoGo(String mode="gogo") {
    bool dataReceived = false;

    Serial.println();
    Serial.println("waiting for data from Go Go Board...");

    if(GoGoConnection){
        while(!dataReceived){
            if(GoGoBoard.isGmessageAvailable("ready-to-train")){
                for(i = 0 ; i < PatternCount ; i++){
                    bool trainingRowReceived = false;
                    while (!trainingRowReceived){
                        if (GoGoBoard.isGmessageAvailable("data-to-train")){
                            // LOG
                            Serial.print("Data-point: ");
                            Serial.print(i+1);
                            Serial.print("/10: ");
                            Serial.print("{ ");
                            for (int j=0; j<InputNodes; j++) {
                                Input[i][j] = convertIntToByteArray(GoGoBoard.Gmessage("data-to-train").toInt())[j];
                                // LOG THE INPUTS HERE:
                                Serial.print(Input[i][j]);
                                Serial.print(", ");
                            }
                            Serial.print("},");
                            trainingRowReceived = true;

                            bool categoryReceived = false;
                                while (!categoryReceived){
                                    if (GoGoBoard.isGmessageAvailable("data-to-train-category")){
                                        int output = GoGoBoard.Gmessage("data-to-train-category").toInt();
                                        Serial.print(" --- category: ");
                                        Serial.println(output);
                                        Target[i][0] = output;
                                        categoryReceived = true;
                                    }
                                }
                        }
                    }
                }  
                dataReceived = true;   
            } 
        }    
    }
}

/***********************************************************************
 * ARDUINO METHODS
***********************************************************************/

void setup(){
    model = false;

    GoGoBoard.begin();
    Serial.begin(115200);

    randomSeed(analogRead(3));
    ReportEvery1000 = 1;
    
    for( p = 0 ; p < PatternCount ; p++ ) {    
        RandomizedIndex[p] = p ;
    }
}  

void loop (){

    if(!GoGoConnection){
        getGoGoConnection();
    }
    if(!model){
        getTrainingDataFromGoGo();
    
        initModel();

        trainModel();

        // When model is trained, continue:
    
        Serial.println ();
        Serial.println(); 
        Serial.print ("TrainingCycle: ");
        Serial.print (TrainingCycle);
        Serial.print ("  Error = ");
        Serial.println (Error, 5);

        toTerminal();

        Serial.println ();  
        Serial.println ();
        Serial.println ("Training Set Solved! ");
        Serial.println ("--------"); 
        Serial.println ();
        Serial.println ();  
        ReportEvery1000 = 1;
    }

    if(Error <= Success){
        model = true;
        predictNew(getDataToPredictFromGoGo());
    }

    return;

}