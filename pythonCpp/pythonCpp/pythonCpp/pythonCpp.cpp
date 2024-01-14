// pythonCpp.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <stdio.h>
#include <conio.h>
#include <Python.h>

int main()
{
	PyObject* pInt;
	if (-1 == _putenv("PYTHONHOME=C:\\Users\\vanderh\\AppData\\Local\\Programs\\Python\\Python311")) {
		printf("putenv failed \n");
		return EXIT_FAILURE;
	}

	//Py_SetPythonHome(L"C:\\Users\\vanderh\\AppData\\Local\\Programs\\Python\\Python311");
 	Py_Initialize();
	PyRun_SimpleString("import sys");
	PyRun_SimpleString("import os");
	PyRun_SimpleString("sys.path.append(os.getcwd())");
	//PyRun_SimpleString("sys.path.append(C:\\Users\\vanderh\\AppData\\Local\\Programs\\Python\\Python311)");
	//PyRun_SimpleString("print('Hello World from Embedded Python!!!')");

	PyObject* myModule = PyImport_ImportModule("something");

	PyObject* myFunction = PyObject_GetAttrString(myModule, (char*)"myabs");
	PyObject* args = PyTuple_Pack(1, PyFloat_FromDouble(-2.0));

	PyObject* myResult = PyObject_CallObject(myFunction, args);
	double result = PyFloat_AsDouble(myResult);

	Py_Finalize();

	printf("\nResult from python:%lf", result);
	printf("\nPress any key to exit...\n");
	if (!_getch()) _getch();
	return 0;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
