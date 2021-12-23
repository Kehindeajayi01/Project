// Creating a quiz app 
#include <iostream>
#include <string>
#include <fstream>
#include <vector>

using namespace std;

const int NUM_OF_QUESTIONS = 5;
const int NUM_OF_OPTIONS = 3;
const int NUM_OF_ANSWERS = 3;

struct Test{
    string questions;
    string options[NUM_OF_OPTIONS];
    int answers;
};

// read in the questions into an array
string printQuestion(Test test[5], int& questionNum){
    ifstream infile;
    int i = 0;
    infile.open("/Users/kehindeajayi01/Desktop/Personal_Learning/C++/Quiz/questions.txt"); // question file 
    while (getline(infile, test[i].questions)){
        i++;
    }
    return test[questionNum].questions;
}

// read in the options using a two dimensional array
string printOption(Test testOption[5], int& questionNum, int& optionNum){
    ifstream infile;
    int i = 0; int j = 0;
    infile.open("/Users/kehindeajayi01/Desktop/Personal_Learning/C++/Quiz/options.txt"); // option file 
    for (int i = 0; i < NUM_OF_QUESTIONS; i++){
        for (int j = 0; j < NUM_OF_OPTIONS; j++){
            infile >> testOption[i].options[j];
        }
    }
    return testOption[questionNum].options[optionNum];
}

// read in the answer into an array
int printAnswer(Test test[5], int& answerNum){
    ifstream infile;
    infile.open("/Users/kehindeajayi01/Desktop/Personal_Learning/C++/Quiz/answers.txt"); // question file 
    for (int i = 0; i < 5; i++){
        infile >> test[i].answers;
    }
    return test[answerNum].answers;
}

int main(){
    
    Test tests[NUM_OF_QUESTIONS];
    Test testOptions[NUM_OF_QUESTIONS];
    int answer, input;
    int score = 0;
    for (int i = 0; i < 5; i++){
        cout << "========================================================" << endl;
        cout << printQuestion(tests, i) << endl;
        for (int j = 0; j < 3; j++){
            
            cout << printOption(testOptions, i, j) << endl;
        }
        answer = printAnswer(tests, i);
        cout << "Choose 1-3: ";
        cin >> input; cout << endl;
        if (input == answer){
            cout << "Correct!" << endl;
            score += 1;
        }
        else 
            cout << "Incorrect!" << endl;
        
    }
    cout << "Your score is: " << score << " / " << NUM_OF_QUESTIONS << endl;
    return 0;
}