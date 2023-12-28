# from django.shortcuts import render
# from django.http import HttpResponse
# from django.core.files.storage import default_storage
# # from backend.generate import runModel
# import asyncio
# import aiohttp
# from django.shortcuts import render
# from django.http import HttpResponse
# from django.core.files.storage import default_storage
# from aiohttp import FormData

# from generate import runModel



# # Create your views here.

# def save_file(file):
#     filepath = default_storage.save('uploads/' + file.name, file)
# def upload_file(request):
#     if request.method == 'POST':
#         uploaded_file = request.FILES['file']
#         # Do something with the uploaded file, for example, save it to the server or process its contents
#         # For demonstration purposes, let's just print the file name
#         print("Uploaded file:", uploaded_file.name)
#         # print()
#         # await asyncio.to_thread(save_file, uploaded_file)
#         filepath=default_storage.save('uploads/'+uploaded_file.name,uploaded_file)
        
#         # content=i=uploaded_file.read()
#         # with open('uploads/'+uploaded_file.name,'rb') as f:
#         #     content=f
#         # content='uploads/'+uploaded_file.name
#         # result=await runModel(content)
#         return HttpResponse('success')
    
#     return render(request, 'index.html')

import asyncio
from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import default_storage
from aiohttp import FormData
from generate import runModel

def MainPage(request):
    return render(request,'index.html') 

def save_file(file):
    filepath = default_storage.save('uploads/' + file.name, file)
    return filepath

async def save_file_async(file):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, save_file, file)

async def upload_file(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['file']
        print("Uploaded file:", uploaded_file.name)
        filepath=save_file(uploaded_file)
        content = 'media/uploads/' + uploaded_file.name
        result = runModel(content)
        # return HttpResponse(result)

    return render(request, 'resultpage.html',{'filename':uploaded_file.name, 'result':result})

def draft(request):
    filename='SAURABH.jpg'
    result='man in blue shirt'
    return render(request, 'resultpage.html',{'filename':filename, 'result':result})
    # return render(request,'index.html')