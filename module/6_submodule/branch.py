#########################################################################################
#########################################################################################
##                                                                                     ##
##                               #### HOW TO IMPORT ####                               ##
##                                                                                     ##
##   import sys                                                                        ##
##   sys.path.append( -> location(dir) of branch.py <- )                               ##
##   import branch as brn                                                              ##
##                                                                                     ##
##   (example -  sys.path.append('C:\\Users\\joo09\\Documents\\GitHub\\branch') )      ##
##                                                                                     ##
##                                                                                     ##
##                              #### HOW TO RE-IMPORT ####                             ##
##                                                                                     ##
##   from imp import reload                                                            ##
##   reload(brn)                                                                       ##
##                                                                                     ##
#########################################################################################
#########################################################################################




#                                   < try this code >



# import sys
# sys.path.append( %%% location(dir) of branch.py %%% )
# import branch as brn
#
# string = '[m] clothes [s] shirts [s] pants [ss] denim [sss] tapered [s] accessories \
#           [m] coffee [s] espresso [s] latte \
#           [m] Python [s] branch [ss] init_theme'
#
# branched_string = brn.branch(string)
# print(branched_string)
#
# themed_str = brn.branch_theme('theme_korean', string)
# print(themed_str)





#########################################################################################
#########################################################################################
####################################<  HOW TO USE  >#####################################
##                                                                                     ##
##                                                                                     ##
##                                     < Input >                                       ##
##                                                                                     ##                    
##  string = '[m] clothes [s] shirts \                                                 ##
##                        [s] pants [ss] denim [sss] tapered \                         ##
##                        [s] accessories \                                            ##
##            [m] coffee  [s] espresso [s] latte \                                     ##
##            [m] Python  [s] branch \                                                 ##
##                        [ss] init_theme'                                             ##
##                                                                                     ##
##                                        *or you can just write down to a single line ##
##                                                                                     ##
##                                                                                     ##
#########################################################################################
##                                                                                     ##
##                                                                                     ##
##                                      < Code >                                       ##
##                                                                                     ##
##                         branched_string = brn.branch(string)                        ##
##                                                                                     ##
##                                                                                     ##
#########################################################################################
##                                                                                     ##
##                                                                                     ##
##                                                                                     ##
##                                     < Result >                                      ##
##                                                                                     ##
##                                      .                                              ##
##                                      ├─clothes                                      ##
##                                      │  ├─shirts                                    ##
##                                      │  ├─pants                                     ##
##                                      │  │  └─denim                                  ##
##                                      │  │     └─tapered                             ##
##                                      │  └─accessories                               ##
##                                      │                                              ##
##                                      ├─coffee                                       ##
##                                      │  ├─espresso                                  ##
##                                      │  └─latte                                     ##
##                                      │                                              ##
##                                      └─Python                                       ##
##                                         └─branch                                    ##
##                                            └─init_theme                             ##
##                                                                                     ##
##                                                                                     ##
##                                                                                     ##
##  *or you can make your own theme                                                    ##
##                                                                                     ##
##                                     < Theme_korean >                                ##
##                                                                                     ##
##                                     .                                               ##
##                                     ㅏㅡclothes                                     ##
##                                     ㅣ  ㅏㅡshirts                                  ##
##                                     ㅣ  ㅏㅡpants                                   ##
##                                     ㅣ  ㅣ  ㄴㅡdenim                               ##
##                                     ㅣ  ㅣ     ㄴㅡtapered                          ##
##                                     ㅣ  ㄴㅡaccessories                             ##
##                                     ㅣ                                              ##
##                                     ㅏㅡcoffee                                      ##
##                                     ㅣ  ㅏㅡespresso                                ##
##                                     ㅣ  ㄴㅡlatte                                   ##
##                                     ㅣ                                              ##
##                                     ㄴㅡPython                                      ##
##                                        ㄴㅡbranch                                   ##
##                                           ㄴㅡtheme_korean                          ##
##                                                                                     ##
##                                                                                     ##
#########################################################################################
#########################################################################################
#########################################################################################


def default() :
    top = '.'
    bar = '│'
    empty = '  '
    middle = '\u251c'
    end = '\u2514'
    tip = '\u2500'

    middle = middle + tip
    end = end + tip
    
    show = 1
    
    return top, bar, empty, middle, end, tip, show


def theme_korean() :
    top = '.'
    bar = 'ㅣ'
    empty = '  '
    middle = 'ㅏ'
    end = 'ㄴ'
    tip = 'ㅡ'

    middle = middle + tip
    end = end + tip
    
    show = 1
    
    return top, bar, empty, middle, end, tip, show

def wave() :
    top = '.'
    bar = '\u3030'
    empty = '  '
    middle = '\u3030'
    end = '\u3030'
    tip = '\u3030'

    middle = middle + tip
    end = end + tip
    
    show = 1
    
    return top, bar, empty, middle, end, tip, show


def larva() :
    top = 'Larva race! \n'
    bar = ' '
    empty = '  '
    middle = '...'
    end = '...'
    tip = '\u0df4'

    middle = middle + tip
    end = end + tip
    
    show = 0

    return top, bar, empty, middle, end, tip, show

#########################################################################################

def branch(string) :
    top, bar, empty, middle, end, tip, show = default()
    bracket_list, string_list = split_string(string)
    br_length = branches_length(bracket_list)
    istip = tip_finder(br_length)
    structure = branch_structure(br_length)
    string_adj = makebranch(top, bar, empty, middle, end, show, string_list, br_length, istip, structure)
    print(string_adj)

    return string_adj

def branch_theme(theme, string) :
    top, bar, empty, middle, end, tip, show = eval(theme + '()')
    bracket_list, string_list = split_string(string)
    br_length = branches_length(bracket_list)
    istip = tip_finder(br_length)
    structure = branch_structure(br_length)
    string_adj = makebranch(top, bar, empty, middle, end, show, string_list, br_length, istip, structure)
    print(string_adj)

    return string_adj


#########################################################################################

def deletestr(characters, string) :
    for x in range(len(characters)) :
        string = string.replace(characters[x], "")

    return string

def unique(list1) :
    unique_list = []
    for x in list1 :
        if x not in unique_list :
            unique_list.append(x)

    return unique_list

def list_to_string(list) :
    string_ints = [str(int) for int in list]
    string = ''.join(string_ints)

    return string



## 1) split strings into '[m] / [s]' and 'strings'
def split_string(string) :
    
    string_copy = string

    bracket_list = []
    string_list = []

    leftbracket = 0
    rightbracket = 0
    end = 0

    while (leftbracket != -1) & (rightbracket != -1) & (end != -1):
        leftbracket = string_copy.find('[')
        rightbracket = string_copy.find(']')

        end = string_copy[leftbracket + 1 :].find('[')
        
        bracket_list.append(deletestr('[]', string_copy[leftbracket : rightbracket + 1]))
        if end == -1 :
            string_list.append(string_copy[rightbracket + 2 : ])
        else :
            string_list.append(string_copy[rightbracket + 2 : end + 1])
        string_copy = string_copy[end + 1 :]
        
    return bracket_list, string_list

## 2) add branches from brackets
def makebranch(top, bar, empty, middle, end, show, string_list, br_length, istip, structure) :
    allstr = []

    for i in range(len(br_length)) :
        
        string_temp = []
        
        # one space for [m]
        if i == 0 :
            allstr.append(top)
        elif i != 0 :
            if br_length[i] == 0 :
                allstr.append(bar)
                
        # detemine tip or middle
        if istip[i] == 0 :
            end_temp = middle
        elif istip[i] == 1:
            end_temp = end
            
        # make branch by structure
        for j, x in enumerate(structure[i]) :
            if j <= br_length[i] :
                if x == str(0) :
                    string_temp.append(' ' + empty)

                elif x == str(1) :
                    if j != len(structure[i]) :
                        if structure[i][j + 1 : ].find(str(1)) == -1 :
                            string_temp.append(end_temp)

                        else :
                            string_temp.append(bar + empty)

                    elif j == len(structure[i]) :
                        string_temp.append(end_temp)
        
        # add original string
        if show == 1 :
            string_temp.append(string_list[i])

        # if i != len(br_length) - 1 :
        #     string_temp.append('\n')

        string_temp_str = ''.join(string_temp)
        allstr.append(string_temp_str)

    length_allstr = []
    for i in range(len(allstr)) :
        length_allstr.append(len(allstr[i]))

    wannabe = (max(length_allstr) // 10 + 1) * 10
    for i in range(len(allstr)) :
        allstr[i] = allstr[i] + ' ' * (wannabe - len(allstr[i])) + '\n'

    return ''.join(allstr)

## 3) branch length
def branches_length(br_list) :
    br_length = []

    for i in range(len(br_list)) :
        if br_list[i] == 'm' :
            br_length.append(0)
        else : 
            br_length.append(len(br_list[i]))
            
    return br_length


## 4) check if tip
def tip_finder(br_length) :
    br_unique = unique(br_length)
    br_length_str = list_to_string(br_length)

    istip = []

    temp_length = br_length_str
    for num, x in enumerate(br_length_str) :
        x1 = temp_length.find(str(x))
        temp_length = temp_length[x1 + 1 :]
        x2 = temp_length.find(str(x))
        
        tip_check = 0
        
        if x2 == -1 :
            tip_check = 1
            istip.append(1)
        else :
            for i, y in enumerate(temp_length[ : x2]) :
                if temp_length[i] < x :
                    tip_check = 1

            if tip_check == 1 :
                istip.append(1)

            else :
                istip.append(0)
    
    return istip

def branch_structure(br_length) :
    br_length_rvs = br_length[:: -1]
    structure = []
    base = []
    for i in range(max(br_length) + 1) :
        base.append(0)
    temp = base

    for i in range(len(br_length_rvs)) :
        temp[br_length_rvs[i]] = 1
        for j in range(len(temp)) :
            if j > br_length_rvs[i] :
                temp[j] = 0
            
            if temp[j] != 0 :
                temp[j] = 1
        structure.append(list_to_string(temp))

    structure = structure[ :: -1]

    return structure