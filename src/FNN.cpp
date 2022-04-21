//Author: Ralph Heymsfeld
//28/06/2018

#include <math.h>
#include <Arduino.h>
#include <GoGoBoardArduino.h>
#include <stdexcept>
#include <ml.h>


int testInt = 0;

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

        GoGoBoard.sendGmessage("arduino-status", "receive-data-to-predict");

        while(!dataReceived){
            if (GoGoBoard.isGmessageAvailable("data-to-predict")){
                dataToPredict = GoGoBoard.Gmessage("data-to-predict").toInt();
                Serial.print("Received: ");
                Serial.println(dataToPredict);
                dataReceived = true;
                GoGoBoard.sendGmessage("arduino-status", "done");  
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

        GoGoBoard.sendGmessage("arduino-status", "receive-data-to-train");

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
                GoGoBoard.sendGmessage("arduino-status", "done");   
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
        RandomizedIndex[p] = p ;                        // Actually not sure what this is for... Does something w.r.t. training the network..
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