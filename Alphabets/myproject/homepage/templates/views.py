from django.http import HttpResponse
from django.shortcuts import render

# Create your views here.
def button(request):
	return render(request,'home.html')
def output(request):
	data="Hello World this is python"
	return render(request,'home.html',{'data':data})
