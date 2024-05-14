# first of all import the socket library 
import socket 
from time import sleep            
 
# next create a socket object 
s = socket.socket()         
print ("Server Socket successfully created")
 
# reserve a port on your computer in our 
# case it is 9999 but it can be anything 
port = 9999    
ipv4 = '10.1.44.125'
 
# Next bind to the port 
# we have not typed any ip in the ip field 
# instead we have inputted an empty string 
# this makes the server listen to requests 
# coming from other computers on the network 

#Trying ipv4
s.bind((ipv4, port))
#s.bind(('', port))         
print ("socket port binded to %s" %(port)) 
print ("socket ipv4 binded to %s" %(ipv4))
 
# put the socket into listening mode 
s.listen(5)     
print ("socket is listening")            
 
# a forever loop until we interrupt it or 
# an error occurs 
while True: 
 
# Establish connection with client. 
  c, addr = s.accept()     
  print ('Got connection from', addr )
 
  # send a thank you message to the client. encoding to send byte type. 
  c.send('Thank you for connecting'.encode()) 




  # for x in range(10000):
  #   try:
  #       c.send(('This is super super super long text, how I expect ASR will be or similar. Some sort of quick redf fox jumped over a log or something.  ' + str(x)).encode()) 
  #   except:
  #       break
  #   sleep(1)
 
  # Close the connection with the client 
  c.close()
   
  # Breaking once connection closed
  break