#include <Arduino.h>

String * GogoMessageTest(String key, String message){

    static String Message[2] = {key, message};

    return Message;

}