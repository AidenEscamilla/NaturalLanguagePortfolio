
import sqlite3
import ssl


class Songs:
    def __init__(self):
        ssl._create_default_https_context = ssl._create_unverified_context
        con = sqlite3.connect("Songs.db")
        con.row_factory = sqlite3.Row

        con.execute("PRAGMA foreign_keys = ON")
        con.execute("CREATE TABLE IF NOT EXISTS Song(url PRIMARY KEY, name, artist, sentiment_Score, Category)")
        con.execute("CREATE TABLE IF NOT EXISTS Song_Not_Found(name, artist, error, url, FOREIGN KEY(url) REFERENCES Song (url))")
        con.execute("CREATE TABLE IF NOT EXISTS Lyrics(lyrics, url, FOREIGN KEY(url) REFERENCES Song(url))")
        con.commit()
        self.connection = con
    
    def __str__(self):
        result = self.connection.execute('SELECT * FROM Song INNER JOIN Lyrics on Song.url = Lyrics.url LIMIT 1')
        row = result.fetchone()
        return str(row['name'], ': ',  row['artist'])
    
    def getAllWithLyrics(self):
        result = self.connection.execute('SELECT * FROM Song INNER JOIN Lyrics on Song.url = Lyrics.url WHERE length(lyrics) > 0')
        rows = result.fetchall()
        return rows
    
    def searchUserSong(self, title):
        half_input = title[:int(len(title)/2)]
        result = self.connection.execute('SELECT * FROM Song WHERE name LIKE ?', ['%'+title+'%'])
        if result == None:
            result = self.connection.execute('SELECT * FROM Song WHERE name LIKE ?',['%'+half_input+'%'])
        rows = result.fetchone()
        
        if rows == None:
            return -1
        else:
            return rows
    
    def searchUserSongAndArtist(self, title, artist):
        half_input = title[:int(len(title)/2)]
        result = self.connection.execute('SELECT * FROM Song WHERE name LIKE ? AND artist LIKE ?', ['%'+title+'%', '%'+artist+'%'])
        if result == None:
            result = self.connection.execute('SELECT * FROM Song WHERE name LIKE ? AND artist LIKE ?',['%'+half_input+'%', '%'+artist+'%'])
        rows = result.fetchone()

        if rows == None:
            return -1
        else:
            return rows

    def getSpecificWithLyrics(self, title, artist):
        result = self.connection.execute('SELECT * FROM Song INNER JOIN Lyrics on Song.url = Lyrics.url WHERE name = ? AND artist = ? AND length(lyrics) > 0', [title, artist])
        rows = result.fetchone()
        return rows
    
    def getCategoryRows(self, category):
        result = self.connection.execute('SELECT * FROM Song WHERE Category = ?', [category])
        rows = result.fetchall()

        if len(rows) == 0:
            return -1
        else:
            return rows
    
    def getRandomRows(self):
        result = self.connection.execute('SELECT * FROM Song ORDER BY RANDOM() LIMIT 5')
        rows = result.fetchall()
        return rows
    
    def insertSong(self, songDict):
        result = self.connection.execute('INSERT OR REPLACE INTO Song(url, name, artist) VALUES(:url, :name, :artist)', songDict)
        self.connection.commit()

    def getAllSongs(self):
        result = self.connection.execute('SELECT * FROM Song')
        rows = result.fetchall()
        return rows
    
    def getTitleArtistSong(self, title, artist):
        result = self.connection.execute('SELECT * FROM Song WHERE name = ? AND artist = ?', [title, artist])
        rows = result.fetchone()
        return rows
    
    def insertToNotFound(self, songNotFound):                                                    #Maybe prob here
        result = self.connection.execute('INSERT OR REPLACE INTO Song_Not_Found VALUES( :name, :artist, :error, :url)', songNotFound)
        self.connection.commit()

    def insertToLyrics(self, lyricsHolder):                                                     #Maybe prob here
        result = self.connection.execute('INSERT OR REPLACE INTO Lyrics VALUES(:lyrics, :url)', lyricsHolder)
        self.connection.commit()