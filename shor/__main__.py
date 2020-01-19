import shor.routines as routines

if __name__ == "__main__":
    # Example from blog post: 35 = 7 * 5
    print(routines.example_single_run(N=35, x=8, random_seed=9, t=12))
    # statistics of algorithm on 35
    routines.example_statistics(N=35, t=12)