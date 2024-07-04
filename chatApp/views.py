from django.views.generic import TemplateView
from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse 
from chat.forms import *

# from neural_network.TESTING_CNN import get
import pyttsx3
import test

from neural_network.TESTING_SVM import get

import chatApp.test

class HomePage(TemplateView):
	template_name = 'index.html'

def info(request):
		return render(request, 'info.html')


def about_us(request):
		return render(request, 'about_us.html')


def DETECTION_PAGE(request):   
    if request.method == 'POST': 
        form = INPUT_IMAGE_FORM(request.POST, request.FILES) 
  
        if form.is_valid(): 
            form.save() 
            return redirect('display_result') 
    else: 
        form = INPUT_IMAGE_FORM() 
    return render(request, 'detection.html', {'form' : form}) 
  

def DETECTION_PAGE_SVM(request):   
    if request.method == 'POST': 
        form = INPUT_IMAGE_FORM(request.POST, request.FILES) 
  
        if form.is_valid(): 
            form.save() 
            return redirect('display_result_SVM') 
    else: 
        form = INPUT_IMAGE_FORM() 
    return render(request, 'detection_SVM.html', {'form' : form}) 


# def predicted_results(request):
# 		if request.method == 'GET':
# 			sym1 =request.GET['filename']
# 			print(sym1)
# 		return render(request, 'detect_result.html')

def display_result(request):   
        IMAGES = INPUT_IMAGES.objects.all()
        print('./' + str(IMAGES[len(IMAGES)-1].Input ))

        bt_result,prevention_action = 1,2

        bt_result = get('./' + str(IMAGES[len(IMAGES)-1].Input ))        


        return render(request, 'display_result.html', {'image':IMAGES[len(IMAGES)-1].Input ,'result':bt_result,'prevention_action':prevention_action})



def display_result_SVM(request):   
        IMAGES = INPUT_IMAGES.objects.all()
        print('./' + str(IMAGES[len(IMAGES)-1].Input ))

        bt_result,prevention_action = 1,2

        bt_result = get('./' + str(IMAGES[len(IMAGES)-1].Input ))        

        return render(request, 'display_result_SVM.html', {'image':IMAGES[len(IMAGES)-1].Input ,'result':bt_result,'prevention_action':prevention_action})
