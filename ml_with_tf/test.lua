#!/usr/local/bin/lua

-- $ wget http://www.lua.org/ftp/lua-5.2.3.tar.gz
-- $ tar zxf lua-5.2.3.tar.gz
-- $ cd lua-5.2.3
-- $ make linux test

print()
print("Hello world !")

print(_VERSION)

io.write("Hello world, from ",_VERSION,"!\n")

a = 1 + 2
print(a)

-- zhushi 
-- Table fields: This is a special type of variable that can hold anything except nil including functions
local a
local b
a= 1
b= 2
print(a)
print(b)

-- print(type a)

g,l = 1, 2
print(g, l)
print("g:",g)
print(type("hello world type"))
a="string !"
print(type(a))
print(type(0))

a = 0
print(type(a))


print(a~=b)


a = 21
b = 10
if( a == b )
then
	print("Line 1 - a is equal to b" )
else
	print("Line 1 - a is not equal to b" )
end


if( a ~= b )
then
	print("Line 2 - a is not equal to b" )
else
	print("Line 2 - a is equal to b" )
end
if ( a < b )
then
	print("Line 3 - a is less than b" )
else
	print("Line 3 - a is not less than b" )
end
if ( a > b )
then
	print("Line 4 - a is greater than b" )
else
	print("Line 5 - a is not greater than b" )
end
-- Lets change value of a and b
a = 5
b = 20
if ( a <= b )
then
	print("Line 5 - a is either less than or equal to b" )
end
if ( b >= a )
then
	print("Line 6 - b is either greater than or equal to b" )
end



a=5
b=20
if (a and b)
	then 
		print("a and b is True")
	end

if (a or b)
	then print("88")
end	

print("Hello ".."Bai yan yan !")


-- # is A unary operator that returns the length of the a string or a table

print(#"hello ")


print("- * - - * - - * - - * - - * - - * - ")
-- while(condition)
-- do
-- 	 statement(s)
-- end
a=10
while (a < 15)
do
print("value of a:", a)
a = a+1
end

print("- * - - * - - * - - * - - * - - * - ")

-- for init,max/min value, increment
-- do
-- statement(s)
-- end

for i=5,-1,-1
do
	print(i)
end


print("- * - - * - - * - - * - - * - - * - ")
-- repeat
-- statement(s)
-- while( condition )


--[ local variable definition --]
a = 10
--[ repeat loop execution --]
repeat
	print("value of a:", a)
	a = a + 1
until( a > 20 )




print("- * - - * - - * - - * - - * - - * - ")
-- one loop inside another loop
-- for loop 

-- for init,max/min value, increment
-- do
-- for init,max/min value, increment
-- do
-- statement(s)
-- end
-- statement(s)
-- end

-- while loop

-- while(condition)
-- do
-- while(condition)
-- do
-- statement(s)
-- end
-- statement(s)
-- end

-- repeat until

-- repeat
-- statement(s)
-- repeat
-- statement(s)
-- until( condition )
-- until( condition )

j =2
for i=2,10 do
	for j=2,(i/j) , 2
		do
			
			if(not(i%j))
			then
				break
			end

			if(j > (i/j))
			then
				print("Value of i is",i)
			-- end
		end
	end
end


print("- * - - * - - * - - * - - * - - * - ")
--[ local variable definition --]
a = 10;
--[ check the boolean condition using if statement --]
if( a < 20 )
then
--[ if condition is true then print the following --]
print("a is less than 20" );
end
print("value of a is :", a);


---function
-- optional_function_scope function function_name( argument1, argument2,
-- argument3..., argumentn)
-- function_body
-- return result_params_comma_separated
-- end


--[[ function returning the max between two numbers --]]
function max(num1, num2)
if (num1 > num2) then
result = num1;
else
result = num2;
end
return result;
end

print(max(1,2))



print("- * - - * - - * - - * - - * - - * - ")
function average(...)
result = 0
local arg={...}
for i,v in ipairs(arg)
	do
		print(i, v)

		result = result + v
end
return result/#arg
end
print("The average is",average(10,5,3,4,5,6))



print("- * - - * - - * - - * - - * - - * - ")
string1 = "Lua"
print("\"String 1 is\"",string1)
string2 = 'Tutorial'
print("String 2 is",string2)
string3 = [["Lua Tutorial"]]
print("String 3 is",string3)


print("- * - - * - - * - - * - - * - - * - ")


string1 = "Lua";
print(string.upper(string1))
print(string.lower(string1))



print("- * - - * - - * - - * - - * - - * - ")

string = "Lua Tutorial"
-- replacing strings
newstring = string.gsub(string,"Tutorial","Language")
print("The new string is",newstring)



print("- * - - * - - * - - * - - * - - * - ")

string = "Lua Tutorial"
-- replacing strings
print(string.find(string,"Tutorial"))
reversedString = string.reverse(string)
print("The new string is",reversedString)



print("- * - - * - - * - - * - - * - - * - ")

string1 = "Lua"
string2 = "Tutorial"
number1 = 10
number2 = 20
-- Basic string formatting
print(string.format("Basic formatting %s %s",string1, string2))
-- Date formatting
date = 2; month = 1; year = 2014
print(string.format("Date formatting %02d/%02d/%03d", date, month,
year))
-- Decimal formatting
print(string.format("%.4f",1/3))

-- Byte conversion
-- First character
print(string.byte("Lua"))
-- Third character
print(string.byte("Lua",3))
-- first character from last
print(string.byte("Lua",-1))
-- Second character
print(string.byte("Lua",2))
-- Second character from last
print(string.byte("Lua",-2))
-- Internal Numeric ASCII Conversion
print(string.char(97))



print("- * - - * - - * - - * - - * - - * - ")

string1 = "Lua"
string2 = "Tutorial"
-- String Concatenations using ..
print("Concatenated string",string1..string2)
-- Length of string
print("Length of string1 is ",string.len(string1))
-- Repeating strings
repeatedString = string.rep(string1,3)
print(repeatedString)


print("- * - - * - - * - - * - - * - - * - ")
array = {"Lua", "Tutorial"}
for i= 0, 2 do
print(array[i])
end
print(array[1])

print("- * - - * - - * - - * - - * - - * - ")


array = {}
for i= -2, 2 do
array[i] = i *2
end






print("- * - - * - - * - - * - - * - - * - ")


-- Initializing the array
array = {}
for i=1,3 do
array[i] = {}
for j=1,3 do
array[i][j] = i*j
end
end
-- Accessing the array
for i=1,3 do
for j=1,3 do
print(array[i][j])
end
end


print("- * - - * - - * - - * - - * - - * - ")

-- Initializing the array
array = {}
maxRows = 3
maxColumns = 3
for row=1,maxRows do
for col=1,maxColumns do
array[row*maxColumns +col] = row*col
end
end
-- Accessing the array
for row=1,maxRows do
for col=1,maxColumns do
print(array[row*maxColumns +col])
end
end




print("- * - - * - - * - - * - - * - - * - ")

array = {"Lua", "Tutorial"}
for key,value in ipairs(array)
do
print(key, value)
end






print("- * - - * - - * - - * - - * - - * - ")

function square(iteratorMaxCount,currentNumber)
if currentNumber<iteratorMaxCount
	then
	currentNumber = currentNumber+1
return currentNumber, currentNumber*currentNumber
	end
end

for i,n in square,3,0
do
	print(i,n)
end


print("- * - - * - - * - - * - - * - - * - ")

function square(iteratorMaxCount,currentNumber)
if currentNumber<iteratorMaxCount
	then
	currentNumber = currentNumber+1
	return currentNumber, currentNumber*currentNumber
	end
end
function squares(iteratorMaxCount)
	return square,iteratorMaxCount,0
end

for i,n in squares(3)
	do
	print(i,n)
end


print("- * - - * - - * - - * - - * - - * - ")

array = {"Lua", "Tutorial"}
function elementIterator (collection)
local index = 0
local count = #collection
-- The closure function is returned
return function ()
index = index + 1
if index <= count
	then
-- return the current element of the iterator
return collection[index]
	end
end
end

for element in elementIterator(array)
	do
	print(element)
end




print("- * - - * - - * - - * - - * - - * - ")

mytable = {}
print("Type of mytable is ",type(mytable))
mytable[1]= "Lua"
mytable["wow"] = "Tutorial"
print("mytable Element at index 1 is ", mytable[1])
print("mytable Element at index wow is ", mytable["wow"])
-- alternatetable and mytable refers to same table
alternatetable = mytable
print("alternatetable Element at index 1 is ", alternatetable[1])
print("mytable Element at index wow is ", alternatetable["wow"])
alternatetable["wow"] = "I changed it"
print("mytable Element at index wow is ", mytable["wow"])
-- only variable released and and not table
alternatetable = nil
print("alternatetable is ", alternatetable)
-- mytable is still accessible
print("mytable Element at index wow is ", mytable["wow"])
mytable = nil
print("mytable is ", mytable)