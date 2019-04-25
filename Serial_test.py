import serial

port = ¡±/dev/ttyAMA0¡å

usart = serial.Serial(port,9600)

usart.flushInput()

 

print (¡°serial test: BaudRate = 9600¡å)

usart.write(¡°please enter the character:\r¡±)

 

while True:

   if( usart.inWaiting()>0 ) :

      receive = usart.read(1)

      print ¡±receive: ¡±,receive

      usart.write(¡°  send: ¡¯¡±)

      usart.write(receive)

      usart.write(¡°¡®\r¡±)