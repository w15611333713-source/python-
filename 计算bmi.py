from unicodedata import category


def calculate_BMI(height,weight):
    BMI = weight/(height*height)
    if BMI < 18.5:
        category = "偏瘦"
    elif BMI <= 25:
        category = "正常"
    elif BMI <= 30:
        category = "偏胖"
    else:category="肥胖"
    print(f"您的BMI分类为：{category}")
    return BMI


result=calculate_BMI(1.79,80)
print(result)
