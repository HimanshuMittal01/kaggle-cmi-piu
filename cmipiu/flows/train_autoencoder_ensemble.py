from metaflow import FlowSpec, step, IncludeFile, Parameter


class PlayListFlow(FlowSpec):
    """
    A flow to help you build your favorite movie playlist.

    The flow performs the following steps:
    1) Ingests a CSV file containing metadata about movies.
    2) Loads two of the columns from the CSV into python lists.
    3) In parallel branches:
       - A) Filters movies by the genre parameter.
       - B) Choose a random movie from a different genre.
    4) Displays the top entries from the playlist.

    """

    movie_data = """movie_title,title_year,genres,gross
Avatar,2009,Action|Adventure|Fantasy|Sci-Fi,760505847
Pirates of the Caribbean: At World's End,2007,Action|Adventure|Fantasy,309404152
Spectre,2015,Action|Adventure|Thriller,200074175
The Dark Knight Rises,2012,Action|Thriller,448130642
John Carter,2012,Action|Adventure|Sci-Fi,73058679
Spider-Man 3,2007,Action|Adventure|Romance,336530303
Tangled,2010,Adventure|Animation|Comedy|Family|Fantasy|Musical|Romance,200807262
Avengers: Age of Ultron,2015,Action|Adventure|Sci-Fi,458991599
Harry Potter and the Half-Blood Prince,2009,Adventure|Family|Fantasy|Mystery,301956980
Batman v Superman: Dawn of Justice,2016,Action|Adventure|Sci-Fi,330249062
Superman Returns,2006,Action|Adventure|Sci-Fi,200069408
Quantum of Solace,2008,Action|Adventure,168368427
"""

    genre = Parameter(
        "genre", help="Filter movies for a particular genre.", default="Sci-Fi"
    )

    recommendations = Parameter(
        "recommendations",
        help="The number of movies to recommend in " "the playlist.",
        default=5,
    )

    @step
    def start(self):
        """
        Parse the CSV file and load the values into a dictionary of lists.

        """
        # For this example, we only need the movie title and the genres.
        columns = ["movie_title", "genres"]

        # Create a simple data frame as a dictionary of lists.
        self.dataframe = dict((column, list()) for column in columns)

        # Parse the CSV header.
        lines = self.movie_data.split("\n")
        header = lines[0].split(",")
        idx = {column: header.index(column) for column in columns}

        # Populate our dataframe from the lines of the CSV file.
        for line in lines[1:]:
            if not line:
                continue

            fields = line.rsplit(",", 4)
            for column in columns:
                self.dataframe[column].append(fields[idx[column]])

        # Compute genre-specific movies and a bonus movie in parallel.
        self.next(self.bonus_movie, self.genre_movies)

    @step
    def bonus_movie(self):
        """
        This step chooses a random movie from a different genre.

        """
        from random import choice

        # Find all the movies that are not in the provided genre.
        movies = [
            (movie, genres)
            for movie, genres in zip(
                self.dataframe["movie_title"], self.dataframe["genres"]
            )
            if self.genre.lower() not in genres.lower()
        ]

        # Choose one randomly.
        self.bonus = choice(movies)

        self.next(self.join)

    @step
    def genre_movies(self):
        """
        Filter the movies by genre.

        """
        from random import shuffle

        # Find all the movies titles in the specified genre.
        self.movies = [
            movie
            for movie, genres in zip(
                self.dataframe["movie_title"], self.dataframe["genres"]
            )
            if self.genre.lower() in genres.lower()
        ]

        # Randomize the title names.
        shuffle(self.movies)

        self.next(self.join)

    @step
    def join(self, inputs):
        """
        Join our parallel branches and merge results.

        """
        # Reassign relevant variables from our branches.
        self.playlist = inputs.genre_movies.movies
        self.bonus = inputs.bonus_movie.bonus

        self.next(self.end)

    @step
    def end(self):
        """
        Print out the playlist and bonus movie.

        """
        print("Playlist for movies in genre '%s'" % self.genre)
        for pick, movie in enumerate(self.playlist, start=1):
            print("Pick %d: '%s'" % (pick, movie))
            if pick >= self.recommendations:
                break

        print("Bonus Pick: '%s' from '%s'" % (self.bonus[0], self.bonus[1]))


if __name__ == "__main__":
    PlayListFlow()