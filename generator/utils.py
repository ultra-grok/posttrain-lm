def extract_parts(original_prompt):
    sub_start = len("SUBREDDIT: ")
    sub_end = original_prompt.find("\nTITLE: ")
    subreddit = original_prompt[sub_start:sub_end].strip()

    title_start = sub_end + len("\nTITLE: ")
    title_end = original_prompt.find("\nPOST: ")
    title = original_prompt[title_start:title_end].strip()

    post_start = title_end + len("\nPOST: ")
    post_end = original_prompt.find("\nTL;DR:")
    post = original_prompt[post_start:post_end].strip()

    return subreddit, title, post

