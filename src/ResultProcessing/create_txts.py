import os as os
import pickle as pickle

# POTENTIAL PROBLEM:
# This script is hardcoded to unpack the "weights".
# FIX: simply add an if-else dependent on the articulation-type and then determine what to index
# from the dictionary.


# This script unpickles results saved as a *.pickle file and creates *.txt files from them.
def main(load_path, save_path):
    # load pickled dictionaries into list of dictionaries
    os.chdir(load_path)
    file_names = os.listdir()
    dicts = []
    for p in file_names:
        file = open(load_path + p, 'rb')
        dicts.append(pickle.load(file))
        file.close()

    # process the dictionaries
    for i in range(len(file_names)):
        file_names[i] = file_names[i].replace('.pickle', '.txt')
    os.chdir(save_path)

    for i in range(len(file_names)):
        f = open(save_path + file_names[i], 'w')
        # fpdf library ignores \t and \n's, so we embedded indentation by adding it manually
        wh = "        "  # an eight-character width whitespace (2xtab)
        f.write(wh + 'Test ID = == ' + str(dicts[i]['test_ID']) + '\n')
        f.write('----------------------------------------------Algorithm parameters----------------------------------------------' + "\n")
        f.write(wh + 'M = == ' + str(dicts[i]['M']) + '\n')
        f.write(wh + 'Step size = == ' + str(dicts[i]['step_size']) + '\n')
        f.write(wh + 'TL max length = == ' + str(dicts[i]['TL_max_length']) + '\n')
        f.write(wh + 'Neighborhood size = == ' + str(dicts[i]['neighborhood_size']) + '\n')
        f.write(wh + 'Max iter = == ' + str(dicts[i]['max_iter']) + '\n')
        f.write(wh + 'Max loops = == ' + str(dicts[i]['max_loops']) + '\n')
        f.write(wh + 'Random seed = == ' + str(dicts[i]['seed_value']) + '\n')
        f.write(wh + 'Weights = {' + "\n")
        for ind, wi in enumerate(dicts[i]['weights']):
            f.write(wh + wh + 'w' + str(ind + 1) + " = == " + str(wi) + "\n")
        f.write(wh + "}\n")
        f.write("----------------------------------------------Performance----------------------------------------------------------\n")
        f.write(wh + 'Time elapsed = == ' + str(dicts[i]['time_elapsed']) + '\n')
        f.write(wh + 'Termination reason = == ' + str(dicts[i]['termination_reason']) + '\n')
        f.write(wh + 'Last iteration = == ' + str(dicts[i]['last_iter']) + '\n')
        f.write("---------------------------------------------------------------------\n")
        f.write(wh + 'Initial solution:' + "\n")
        f.write(wh + wh + 'x = { ' + "\n")
        for ind, xi in enumerate(dicts[i]['init_sol'].get_x()):
            f.write(wh + wh + wh + 'x' + str(ind + 1) + " = == " + str(xi) + "\n")
        f.write(wh + wh + '}\n')
        f.write(wh + wh + 'f = {\n')
        for ind, fi in enumerate(dicts[i]['init_sol'].get_y()):
            f.write(wh + wh + wh + 'f' + str(ind + 1) + " = == " + str(fi) + "\n")
        f.write(wh + wh + '}\n')
        f.write(wh + wh + 'unicriteria value = == ' + str(dicts[i]['init_sol'].get_val()) + "\n")
        f.write("---------------------------------------------------------------------\n")
        f.write(wh + 'Best solution:\n')
        f.write(wh + wh + 'x = {\n')
        for ind, xi in enumerate(dicts[i]['global_best_sol'].get_x()):
            f.write(wh + wh + wh + 'x' + str(ind + 1) + " = == " + str(xi) + "\n")
        f.write(wh + wh + '}\n')
        f.write(wh + wh + 'f = {\n')
        for ind, fi in enumerate(dicts[i]['global_best_sol'].get_y()):
            f.write(wh + wh + wh + 'f' + str(ind + 1) + " = == " + str(fi) + "\n")
        f.write(wh + wh + '}\n')
        f.write(wh + wh + 'unicriteria value = == ' + str(dicts[i]['global_best_sol'].get_val()) + "\n")
        f.write("---------------------------------------------------------------------\n")
        f.write(wh + 'Last solution:\n')
        f.write(wh + wh + 'x = {\n')
        for ind, xi in enumerate(dicts[i]['curr_sol'].get_x()):
            f.write(wh + wh + wh + 'x' + str(ind + 1) + " = == " + str(xi) + "\n")
        f.write(wh + wh + '}\n')
        f.write(wh + wh + 'f = {\n')
        for ind, fi in enumerate(dicts[i]['curr_sol'].get_y()):
            f.write(wh + wh + wh + 'f' + str(ind + 1) + " = == " + str(fi) + "\n")
        f.write(wh + wh + '}\n')
        f.write(wh + wh + 'unicriteria value = == ' + str(dicts[i]['curr_sol'].get_val()))
        f.close()


if __name__ == '__main__':
    #main('/home/kemal/Programming/Python/Articulation/data/pickles/aposteriori/BK1/', '/home/kemal/Programming/Python/Articulation/data/txts_and_plots/aposteriori/BK1/txts/')
    #main('/home/kemal/Programming/Python/Articulation/data/pickles/aposteriori/IM1/', '/home/kemal/Programming/Python/Articulation/data/txts_and_plots/aposteriori/IM1/txts/')
    #main('/home/kemal/Programming/Python/Articulation/data/pickles/aposteriori/SCH1/', '/home/kemal/Programming/Python/Articulation/data/txts_and_plots/aposteriori/SCH1/txts/')
    main('/home/kemal/Programming/Python/Articulation/data/pickles/aposteriori/FON/', '/home/kemal/Programming/Python/Articulation/data/txts_and_plots/aposteriori/FON/txts/')
    #main('/home/kemal/Programming/Python/Articulation/data/pickles/aposteriori/TNK/', '/home/kemal/Programming/Python/Articulation/data/txts_and_plots/aposteriori/TNK/txts/')
    #main('/home/kemal/Programming/Python/Articulation/data/pickles/aposteriori/OSY/', '/home/kemal/Programming/Python/Articulation/data/txts_and_plots/aposteriori/OSY/txts/')



