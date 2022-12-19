import os
import shutil

def network_share_auth(share, username, password, drive_letter='P'):

    cmd_parts = ["NET USE %s: %s" % (drive_letter, share)]

    if password:

        cmd_parts.append(password)

    if username:

        cmd_parts.append("/USER:%s" % username)

    os.system(" ".join(cmd_parts))

    shutil.copyfile("Test.exe", r"C:Test.exe")

    try:

        yield

    finally:

        os.system("NET USE %s: /DELETE" % drive_letter)


network_share_auth();