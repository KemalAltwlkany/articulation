import os as os
import pickle as pickle

# POTENTIAL PROBLEM:
# This script is hardcoded to unpack the "weights".
# FIX: simply add an if-else dependent on the articulation-type and then determine what to index
# from the dictionary.


# This script unpickles results saved as a *.pickle file and creates *.txt files from them.
def main(art_type='aposteriori', load_path=None, save_path=None):
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

        # Update 10.08.2020. - this is the only difference between a priori and a posteriori
        if art_type == 'aposteriori':
            f.write(wh + 'Weights = {' + "\n")
            for ind, wi in enumerate(dicts[i]['aspirations']):
                f.write(wh + wh + 'z' + str(ind + 1) + " = == " + str(wi) + "\n")
            f.write(wh + "}\n")
        elif art_type == 'apriori':
            f.write(wh + 'Aspiration levels = {' + "\n")
            for ind, zj in enumerate(dicts[i]['aspirations']):
                f.write(wh + wh + 'z' + str(ind + 1) + " = == " + str(zj) + "\n")
            f.write(wh + "}\n")

        if art_type == 'progressive':
            f.write(wh + 'Total repetitions = ==' + str(dicts[i]['total_reps']) + '\n')


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
    # A POSTERIORI TESTS
    #main(art_type='aposteriori', load_path='/home/kemal/Programming/Python/Articulation/data/pickles/aposteriori/BK1/', save_path='/home/kemal/Programming/Python/Articulation/data/txts_and_plots/aposteriori/BK1/txts/')
    #main(art_type='aposteriori', load_path='/home/kemal/Programming/Python/Articulation/data/pickles/aposteriori/IM1/', save_path='/home/kemal/Programming/Python/Articulation/data/txts_and_plots/aposteriori/IM1/txts/')
    #main(art_type='aposteriori', load_path='/home/kemal/Programming/Python/Articulation/data/pickles/aposteriori/SCH1/', save_path='/home/kemal/Programming/Python/Articulation/data/txts_and_plots/aposteriori/SCH1/txts/')
    #main(art_type='aposteriori', load_path='/home/kemal/Programming/Python/Articulation/data/pickles/aposteriori/FON/', save_path='/home/kemal/Programming/Python/Articulation/data/txts_and_plots/aposteriori/FON/txts/')
    #main(art_type='aposteriori', load_path='/home/kemal/Programming/Python/Articulation/data/pickles/aposteriori/TNK/', save_path='/home/kemal/Programming/Python/Articulation/data/txts_and_plots/aposteriori/TNK/txts/')
    #main(art_type='aposteriori', load_path='/home/kemal/Programming/Python/Articulation/data/pickles/aposteriori/OSY/', save_path='/home/kemal/Programming/Python/Articulation/data/txts_and_plots/aposteriori/OSY/txts/')

    # A PRIORI TESTS
    #main(art_type='apriori', load_path='/home/kemal/Programming/Python/Articulation/data/pickles/apriori/BK1/', save_path='/home/kemal/Programming/Python/Articulation/data/txts_and_plots/apriori/BK1/txts/')
    #main(art_type='apriori', load_path='/home/kemal/Programming/Python/Articulation/data/pickles/apriori/IM1/', save_path='/home/kemal/Programming/Python/Articulation/data/txts_and_plots/apriori/IM1/txts/')
    #main(art_type='apriori', load_path='/home/kemal/Programming/Python/Articulation/data/pickles/apriori/SCH1/', save_path='/home/kemal/Programming/Python/Articulation/data/txts_and_plots/apriori/SCH1/txts/')
    #main(art_type='apriori', load_path='/home/kemal/Programming/Python/Articulation/data/pickles/apriori/FON/', save_path='/home/kemal/Programming/Python/Articulation/data/txts_and_plots/apriori/FON/txts/')
    #main(art_type='apriori', load_path='/home/kemal/Programming/Python/Articulation/data/pickles/apriori/TNK/', save_path='/home/kemal/Programming/Python/Articulation/data/txts_and_plots/apriori/TNK/txts/')
    #main(art_type='apriori', load_path='/home/kemal/Programming/Python/Articulation/data/pickles/apriori/OSY/', save_path='/home/kemal/Programming/Python/Articulation/data/txts_and_plots/apriori/OSY/txts/')

    # PROGRESSIVE TESTS
    main(art_type='progressive', load_path='/home/kemal/Programming/Python/Articulation/data/pickles/progressive/BK1/', save_path='/home/kemal/Programming/Python/Articulation/data/txts_and_plots/progressive/BK1/txts/')
    #print('Not active')


