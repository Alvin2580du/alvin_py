import pandas as pd


def get_counts(sequence):
    # 对一个列表统计频率，出现就+1  列表， 字典， 集合  [] set()   {'a':{'a':1}, 'b':2}
    counts = {}
    for x in sequence:
        if x in counts:
            counts[x] += 1
        else:
            counts[x] = 1
    return counts


def name(f, m, l):
    return "{} {} {}".format(f, m, l)


def build_one():
    """
    1、找到相对应的样本记录的方法： 首先把first_name、middle_name、last_name三个字段合并为一个字段name
    然后根据相同ssn、name还有gender判断是不是用一个样本，如果这三列相同，则判定为是同一个records。
    2、emp-edu 中有4075条记录是其独有的，medical中的ssn全部在emp-edu中出现，我们将这部分记录直接删除，避免合并数据后出现缺失值。
    3、数据存存在9123条重复records，对于重复records需要删除，已经做了删除处理。
    """
    emp_edu_data = pd.read_csv("emp-edu.csv")
    emp_edu_data['name'] = emp_edu_data.apply(lambda row: name(row['first_name'], row['middle_name'], row['last_name']),
                                              axis=1)
    del emp_edu_data['first_name']
    del emp_edu_data['middle_name']
    del emp_edu_data['last_name']

    emp_edu_col = emp_edu_data.columns.tolist()
    print(len(emp_edu_col), emp_edu_data.shape)
    emp_edu_data_ssn = emp_edu_data['ssn'].values.tolist()

    medical_data = pd.read_csv("medical.csv")
    medical_data['name'] = medical_data.apply(lambda row: name(row['first_name'], row['middle_name'], row['last_name']),
                                              axis=1)
    del medical_data['first_name']
    del medical_data['middle_name']
    del medical_data['last_name']
    medical_col = medical_data.columns.tolist()
    print(len(medical_col), medical_data.shape)
    medical_data_ssn = medical_data['ssn'].values.tolist()
    ssn_1 = [i for i in emp_edu_data_ssn if i not in medical_data_ssn]
    print(len(ssn_1))
    ssn_2 = [i for i in medical_data_ssn if i not in emp_edu_data_ssn]
    print(len(ssn_2))
    ssn_all = ssn_1 + ssn_2
    print(len(ssn_all), ssn_all)

    signle = pd.merge(emp_edu_data, medical_data, on=['ssn', 'name', 'gender'])
    print(type(signle))
    print(signle.shape)
    signle_drop = signle[~signle['ssn'].isin(ssn_all)]
    print(signle_drop.shape, '222222222222222')
    signle_dup = signle_drop.drop_duplicates(subset='ssn', keep='last')
    print(signle_dup.shape)
    print(signle_drop.shape[0] - signle_dup.shape[0])
    signle_dup.to_csv("single.csv", index=None)


def replace_phone(input):
    if " " in input:
        return input
    else:
        return '-1'


def replace_bmi(input):
    try:
        if int(input) < 0:
            return "-1"
        else:
            return input
    except:
        return "-1"


def build_partTwo():
    """
    1、存在缺失值，共有14个指标存在缺失数据，缺失数据的个数为49211个。因为此数据集为个人信息数据，无法使用0或者均值等数值方法来代替缺失值，
    所以使用“-1" 来填充缺失数据。
    2、数据中的phone字段，存在一些非法的电话号码，比如邮箱等。对于这类数据，需要替换为缺失数据，并用“-1”代替。还有bmi的值存在负数，
    我们知道，bmi的值肯定是正数， 不可能是负数。所以需要把bmi的值为负数的部分替换为“-1”，数据类型为字符串，不可参与数学运算。
    """
    signle_one = pd.read_csv("single.csv")
    num_na = signle_one.count()

    num_na_df = pd.DataFrame(num_na)
    num_na_df['na'] = signle_one.shape[0] - num_na_df[0]
    # num_na_df.to_csv("num_na.csv")
    k = 0
    c = 0
    for columname in signle_one.columns:
        if signle_one[columname].count() != len(signle_one):
            c += 1
            loc = signle_one[columname][signle_one[columname].isnull().values == True].index.tolist()
            k += len(loc)
            print('missing value location: col：{}, rows num: ####{}####, rows:{}'.format(columname, len(loc), loc))
    print("{},{}".format(c, k))
    signle_one_fillna = signle_one.fillna('-1')
    signle_one_fillna['phone'] = signle_one_fillna['phone'].apply(replace_phone)
    print(signle_one_fillna['phone'])
    signle_one_fillna['bmi'] = signle_one_fillna['bmi'].apply(replace_bmi)
    signle_one_fillna.to_csv("single_one_fillna.csv", index=None)


def get_phone(p1, p2):
    if p2 == '-1':
        return p1
    if p1 == '-1':
        return p2
    if p1 != p2 and p1 != "-1" and p2 != '-1':
        return "{}|{}".format(p1, p2)


def build_partThree():
    """
    1、汇总的数据里面包含phone_med和phone两列，分别来源于emp-edu.csv和medical.csv， 对于同一个人的电话号码存在不一致的情况，
    我们把phone缺失的号码用phone_med 来代替，把phone_med中缺失的号码用phone来代替。如果phone_med和phone都存在的情况，
    我们保留两个电话号码，并用竖线“|”分割开来。这样做的好处是保证了数据不丢失。另外还有street_address_med	suburb_med
    postcode_med state_med直接删除，因为这几个变量是具有实际意义的变量，一般不会有变化。
    2、考虑教育和职业与健康的关系，我们建议删除无关变量。比如street_address、suburb、postcode、credit_card_number。
        
    """
    data = pd.read_csv("single_one_fillna.csv")
    data['phone_num'] = data.apply(lambda row: get_phone(row['phone_med'], row['phone']), axis=1)
    del data['phone_med']
    del data['phone']
    del data['street_address_med']
    del data['suburb_med']
    del data['postcode_med']
    del data['state_med']
    # 删除无关变量，reduction data
    del data['street_address']
    del data['suburb']
    del data['postcode']
    del data['credit_card_number']
    data.to_csv("1single_final.csv", index=None)
    print(data.shape)

if __name__ == '__main__':
    method = 'build_one'

    # for method in ['build_one', 'build_partTwo', 'build_partThree']:

    if method == 'build_one':
        build_one()

    if method == 'build_partTwo':
        build_partTwo()

    if method == 'build_partThree':
        build_partThree()


