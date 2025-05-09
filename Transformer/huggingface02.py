from transformers import pipeline

generator = pipeline(task="text-generation")
text01 = generator("Three Rings for the Elven-kings under the sky, Seven for the Dwarf-lords in their halls of stone")

text02 = generator ("The original theory of relativity is based upon the premise that all coordinate systems in relative uniform translatory motion to each other are equally valid and equivalent ")

text03 = generator ("It takes a great deal of bravery to stand up to our enemies")

print("> 01 : ", text01)
print("> 02 : ", text02)
print("> 03 : ", text03)