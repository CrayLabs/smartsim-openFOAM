import numpy as np
from os import listdir, getcwd

def replaceZeroes(data):
    min_nonzero = np.min(data[np.nonzero(data)])
    data[data == 0] = min_nonzero
    return data

# Change the line below, based on U file
# Foundation users set it to 20, ESI users set it to 21
LINE = 20

def read_scalar(filename):
    # Read file
    file = open(filename,'r')
    lines_1 = file.readlines()
    file.close()

    num_cells_internal = int(lines_1[LINE].strip('\n'))
    lines_1 = lines_1[LINE+2:LINE+2+num_cells_internal]

    for i in range(len(lines_1)):
        lines_1[i] = lines_1[i].strip('\n')

    field = np.asarray(lines_1).astype('double').reshape(num_cells_internal,1)
    field = replaceZeroes(field)

    return field


def read_vector(filename): # Only x,y components
    file = open(filename,'r')
    lines_1 = file.readlines()
    file.close()

    num_cells_internal = int(lines_1[LINE].strip('\n'))
    lines_1 = lines_1[LINE+2:LINE+2+num_cells_internal]

    for i in range(len(lines_1)):
        lines_1[i] = lines_1[i].strip('\n')
        lines_1[i] = lines_1[i].strip('(')
        lines_1[i] = lines_1[i].strip(')')
        lines_1[i] = lines_1[i].split()

    field = np.asarray(lines_1).astype('double')[:,:2]

    return field

def get_last_time_step(dir):
    """ This function returns the name of the last
    time step directory.

    :param dir: The directory containing all of the timestep
                information
    :type dir: str
    :return: The name of the subdirectory containing the last time step
    :rtype: str

    """

    return str(max([fd for fd in listdir(dir) if fd.isnumeric()]))

if __name__ == '__main__':
    print('Velocity reader file')

    heights = [2.0, 1.5, 0.5, 0.75, 1.75, 1.25]

    total_dataset = []

    # Read Cases
    for i, h in enumerate(heights, start=1):
        print(f"case{i}")
        end_time = "0"
        file_dir = "/".join([getcwd(),f"Case{i}",end_time])
        U = read_vector("/".join([file_dir, "U"]))
        print(U.shape)

        end_time = get_last_time_step("/".join([getcwd(),f"Case{i}"]))
        file_dir = "/".join([getcwd(),f"Case{i}",end_time])
        nut = read_vector("/".join([file_dir, "nut"]))
        print(nut.shape)

        file_dir = "/".join([getcwd(),f"Case{i}","constant"])
        cx = read_vector("/".join([file_dir, "cx"]))
        cy = read_vector("/".join([file_dir, "cy"]))
        print(cx.shape)
        print(cy.shape)

        h = np.ones(shape=(np.shape(U)[0],1),dtype='double') * h
        print(h)

        temp_dataset = np.concatenate((U,cx,cy,h,nut),axis=-1)
        total_dataset.append(temp_dataset)

    total_dataset = np.concatenate(total_dataset)

    # Save data
    np.save('Total_dataset.npy',total_dataset)

    # Save the statistics of the data

    means = np.mean(total_dataset,axis=0).reshape(1,np.shape(total_dataset)[1])
    stds = np.std(total_dataset,axis=0).reshape(1,np.shape(total_dataset)[1])

    # Concatenate
    op_data = np.concatenate((means,stds),axis=0)

    # Save data to file
    total_text = ""

    header = " ".join([str(c) for c in op_data.shape])
    header += '\n'
    total_text += header + '(' + '\n'

    data_line_1 = " ".join([str(f) for f in op_data[0]])
    total_text += '(' + data_line_1 + ')\n'

    data_line_2 = " ".join([str(f) for f in op_data[1]])
    total_text += '(' + data_line_2 + ')\n' + ');' + '\n'

    with open('means', 'w') as f:
        f.write(total_text)

    print(total_text)

    # Need to write out in OpenFOAM rectangular matrix format

    print('Means:')
    print(means)
    print('Stds:')
    print(stds)
