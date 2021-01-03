#!/usr/bin/env python
# coding: utf-8



def get_mat_sparsity(ratings):

    count_nonzero = ratings.select("rating").count()


    total_elements = ratings.select("userId").distinct().count() * ratings.select("movieId").distinct().count()


    sparsity = (1.0 - (count_nonzero *1.0)/total_elements)*100
    print("The ratings dataframe is ", "%.2f" % sparsity + "% sparse.")

