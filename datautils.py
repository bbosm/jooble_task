'''
utils for data input/output
'''

class TsvFileReader:
    def __init__(self, path, separator='\t'):
        self.path = path
        self.separator = separator
        self.f = None
        self.columns = []

    def __del__(self):
        self.close()

    def _readline(self):
        line = self.f.readline()
        while len(line) > 0:
            yield line[ :-1]
            line = self.f.readline()

    def is_open(self):
        return self.f and not(self.f.closed)

    def open(self):
        if self.is_open():
            raise UserWarning('Close file {} before open()'.format(self.path))

        self.f = open(self.path, 'r')
        try:
            self.columns = next(self._readline()).split(self.separator)
        except StopIteration:
            raise EOFError('Empty file {}'.format(self.path))

    def close(self):
        if self.is_open():
            self.f.close()

    def iterrows(self): # row is dict(column = value)
        if not self.is_open():
            raise UserWarning('Use open() of {} before row_generator()'.format(self.path))

        for i, line in enumerate(self._readline()):
            values = line.split(self.separator)
            if len(values) != len(self.columns):
                raise ValueError('Wrong row in {}, line {}: "{}"'.format(self.path, i, line))
            yield { column : value for column, value in zip(self.columns, values) }
        self.f.close()


class TsvFileWriter:
    def __init__(self, path, columns, separator='\t'):
        self.path = path
        self.columns = columns
        self.separator = separator
        self.f = None

    def __del__(self):
        self.close()

    def is_open(self):
        return self.f and not(self.f.closed)

    def _write_columns(self):
        if not self.is_open():
            raise UserWarning('Use open() of {} before _write_columns()'.format(self.path))
        self.f.write(self.separator.join(self.columns))
        self.f.write('\n')

    def write_row(self, row):
        if not self.is_open():
            raise UserWarning('Use open() of {} before write_row(row)'.format(self.path))
        self.f.write(self.separator.join(str(row[key]) for key in self.columns))
        self.f.write('\n')

    def open(self):
        if self.is_open():
            raise UserWarning('Close file {} before open()'.format(self.path))
        self.f = open(self.path, 'w')
        self._write_columns()

    def close(self):
        if self.is_open():
            self.f.close()