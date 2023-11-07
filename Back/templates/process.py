print("HTTP/1.0 200 OK\n")
import cgi
form = cgi.FieldStorage()
f_name=form["name"].value
s_name=form["Gender"].value
r1=form["r1"].value
my_class=form["class"].value


print("<br><b>First Name</b>",f_name)
print("<br><b>Second Name</b>",s_name)
