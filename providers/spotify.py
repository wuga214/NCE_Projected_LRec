import numpy as np
from scipy.sparse import csr_matrix


def getArtistMatrix(ratings, artist):

    m, n = ratings.shape
    num_artists = len(artist)
    artist_matrix = csr_matrix((m, num_artists), dtype=np.int8)

    for i in xrange(m):
        song_index = ratings[i].nonzero()[1]
        artists_in_row = set(np.take(artist, song_index))
        num_artists_in_row = len(artists_in_row)
        data = np.ones(num_artists_in_row)
        row_index = np.full(num_artists_in_row, i)
        artist_matrix = artist_matrix + csr_matrix((data, (row_index, num_artists)),
                                                   shape=(m, num_artists),
                                                   dtype=np.int8)

    return artist_matrix
