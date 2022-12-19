import lab_cheat


def doc_of_function(function, cls: bool = False):
    """
    Makes a mini html file from the documentation of the function
    :param function: the function from which documentation we are making html file
    :param cls: parameter indicating whether the function belongs to any class
    """
    doc_file = open('doc_file.txt', 'w+')
    doc_file.write(function.__doc__)
    doc_file.close()
    if cls:
        shift = 4
    else:
        shift = 0

    with open('doc_file.txt', 'r') as file:
        doc_list = file.readlines()
        description = ''
        values = []
        k = 0
        for i in range(1, len(doc_list) - 1):
            if doc_list[i][shift + 4] != ':':
                description += doc_list[i]
            elif doc_list[i][shift + 4] == ':':
                k += 1
                values.append(doc_list[i].split(':'))
        for j in range(k):
            values[j][1] = values[j][1].capitalize()
            values[j][2] = values[j][2][1:]
        description = description[shift + 4:len(description) - 1]

    with open('new.html', 'w') as file:
        if description == '':
            file.write(description)
        else:
            file.write('<p>{}</p>\n'.format(description))
        for i in range(k):
            file.write('<dl class="field-list simple">\n')
            file.write('<dt class="field-odd">{}<span class="colon">:</span></dt>\n'.format(values[i][1]))
            file.write('<dd class="field-odd"><p>{}</p>\n'.format(values[i][2]))
            file.write('</dd>\n')
            file.write('</dl>\n')
        file.write('\n...................................................................................\n')

        for i in range(1, len(doc_list)-1):
            string = doc_list[i].replace('\n', '')
            file.write('<span class="sd">{}</span>\n'.format(string))


doc_of_function(lab_cheat.plot.mnk_through0)
