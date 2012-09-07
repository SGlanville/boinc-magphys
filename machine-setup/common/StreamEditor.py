"""
A simple Stream Editor
"""
import logging

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)-15s:' + logging.BASIC_FORMAT)

class StreamEditor:

    def __init__(self):
        self._substitutions = {}

    def substitute(self, start, end = '', to=''):
        """Add a substitution

        :param start: the text to start the substitution on
        :param end: if present the end of a text block to substitute on
        :param to: the text to substitute with
        """
        self._substitutions[start] = [end, to]

    def _find_start(self, text, start):
        index = -1
        key = None
        for starts in self._substitutions.iterkeys():
            index1 = text.find(starts, start)
            if index1 >= 0:
                if index == -1:
                    index = index1
                    key = starts
                elif index1 == index and len(starts) > len(key):
                    index = index1
                    key = starts
                elif index1 < index:
                    index = index1
                    key = starts

        return index, key

    def __call__(self, text):
        """
        Walk through the text performing the replaces required.

        :param text: the text to work on

        >>> a = StreamEditor()
        >>> a.substitute('does not match',to='new')
        >>> a('turn old to new')
        'turn old to new'

        >>> a = StreamEditor()
        >>> a.substitute('old',to='new')
        >>> a('turn old to new')
        'turn new to new'


        >>> a = StreamEditor()
        >>> a.substitute('old',to='new',end='to')
        >>> a('turn old to new')
        'turn new new'

        >>> a = StreamEditor()
        >>> a.substitute('old',to='new',end='to')
        >>> a('''turn old
        ... 123456
        ... to
        ... abc new''')
        'turn new\\nabc new'

        """
        index = 0
        new_list = []
        while index < len(text):
            index1, match = self._find_start(text, index)
            if index1 >= 0:
                for i in range(index, index1):
                    new_list.extend(text[i])

                tuple = self._substitutions[match]
                if tuple[0] == '':
                    # A straight substitution
                    index = index1 + len(match)
                    new_list.extend(list(tuple[1]))
                else:
                    # Need to find the end
                    index2 = text.find(tuple[0], index1 + len(match))
                    if index2 >= 0:
                        index = index2 + len(tuple[0])
                        new_list.extend(list(tuple[1]))
                    else:
                        # Could find the end so just ad the text
                        index += len(match)
                        new_list.extend(list(match))
            else:
                # No more matches
                for i in range(index, len(text)):
                    new_list.append(text[i])
                index = len(text)

        return ''.join(new_list)


if __name__ == "__main__":
    import doctest
    doctest.testmod()