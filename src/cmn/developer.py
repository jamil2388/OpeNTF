

class Developer(object):

    def __init__(self, name: str, id: int, url: str, is_owner: bool = False):
        """

        Args:
            name: name of the developer
            id: the id of his/her GitHub account
            profile_url: the url to the developer's profile
            contributions_count: number of contributions made

        Returns:
            None
        """
        self.name = name
        self.id = id
        self.is_owner = is_owner
        self.url = url
